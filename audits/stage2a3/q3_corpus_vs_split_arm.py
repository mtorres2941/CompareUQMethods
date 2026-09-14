"""Does the active corpus still match the empirical arm after the split?

The criterion for reopening generation, set by the Stage 2a-3 prompt. The
corpus is scored against the UNSPLIT and the SPLIT arm with the existing tuning
objective, every characteristic weighted equally, and the movement is compared
against the seed-to-seed noise measured in audits/stage2a2/p10_config_noise.py:
objective standard deviation 0.0066, mode-count total variation 0.029.

If the movement sits inside that noise, the calibration survives the split and
nothing is regenerated.

    conda run -n compareuq python audits/stage2a3/q3_corpus_vs_split_arm.py [label]
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'audits', 'stage2a'))
sys.path.insert(0, os.path.join(ROOT, 'audits', 'stage2a2'))

import coverage  # noqa: E402
import empirical  # noqa: E402
import modality as MD  # noqa: E402
from customstats import empirical_metadata  # noqa: E402
from b5_tune_configuration import WEIGHTS, MODE_TV_WEIGHT  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2a3')
SEED = 42

#: audits/stage2a2/p10_config_noise.py, one configuration over repeated seeds.
NOISE_OBJECTIVE_SD = 0.0066
NOISE_MODE_TV = 0.029


def arm(split):
    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0],
                              split=split)
    met = pd.DataFrame({m: empirical_metadata(x, w)
                        for m, (x, w) in ds.items()}).T.astype(float)
    met.index.name = 'material'
    rng = np.random.default_rng(0)
    modes = np.array([MD.n_modes_silverman(x, rng=rng, nboot=100)
                      for x, _ in ds.values()])
    vis = np.array([MD.n_modes_visible(x) for x, _ in ds.values()])
    return met.reset_index(), modes, vis


def objective(emp_met, emp_modes, emp_vis, syn_met, syn_modes, syn_vis):
    d = coverage.distribution_comparison(emp_met, syn_met)
    d['weight'] = d.metric.map(WEIGHTS).fillna(1.0)
    tv = sum(abs(float((emp_modes == k).mean()) - float((syn_modes == k).mean()))
             for k in range(1, 7)) / 2.0
    vtv = sum(abs(float((emp_vis == k).mean()) - float((syn_vis == k).mean()))
              for k in range(1, 7)) / 2.0
    num = float((d.w1_standardized * d.weight).sum()) + MODE_TV_WEIGHT * (tv + vtv)
    den = float(d.weight.sum()) + 2 * MODE_TV_WEIGHT
    return d, dict(weighted_objective=num / den,
                   mean_w1_unweighted=float(d.w1_standardized.mean()),
                   mode_tv=tv, visible_tv=vtv,
                   unimodal_empirical=float((emp_modes == 1).mean()),
                   unimodal_synthetic=float((syn_modes == 1).mean()),
                   visible_unimodal_empirical=float((emp_vis == 1).mean()),
                   visible_unimodal_synthetic=float((syn_vis == 1).mean()),
                   worst_metric=d.iloc[0].metric,
                   worst_w1=float(d.iloc[0].w1_standardized))


def corpus_modality(label):
    """Mode counts for the corpus, cached: 10,000 bootstraps take about 25 min.

    The cache is keyed on the corpus label and holds only counts derived from
    the corpus's own values, which never change once a corpus is written.
    """
    cache = os.path.join(TABLES, f'CACHE_modality_{label}.npz')
    met, _ = corpus_stats.__wrapped__(label) if False else (None, None)
    met = pd.read_parquet(os.path.join(ROOT, 'data', 'processed',
                                       f'corpus_{label}', 'metrics.parquet'))
    met = met[~met.is_probe]
    if os.path.exists(cache):
        z = np.load(cache)
        return met, z['modes'], z['vis']
    vals = pd.read_parquet(os.path.join(ROOT, 'data', 'processed',
                                        f'corpus_{label}', 'values.parquet'))
    ids = set(met.dataset.astype(str))
    rng = np.random.default_rng(1)
    modes, vis = [], []
    for ds, g in vals.groupby('dataset_id', observed=True):
        if str(ds) not in ids:
            continue
        x = g['value'].to_numpy()
        if len(x) >= 4:
            modes.append(MD.n_modes_silverman(x, rng=rng, nboot=60))
            vis.append(MD.n_modes_visible(x))
    modes, vis = np.array(modes), np.array(vis)
    np.savez(cache, modes=modes, vis=vis)
    return met, modes, vis


def main(label):
    os.makedirs(TABLES, exist_ok=True)
    met, modes, vis = corpus_modality(label)
    print(f'corpus {label}: {len(met):,} datasets\n', flush=True)

    rows, detail = [], []
    for split in [False, True]:
        emet, emodes, evis = arm(split)
        name = f'{"split" if split else "unsplit"} ({len(emet)})'
        d, s = objective(emet, emodes, evis, met, modes, vis)
        rows.append(dict(empirical_arm=name, n_datasets=len(emet), **s))
        detail.append(d.assign(empirical_arm=name))
    out = pd.DataFrame(rows)
    out.insert(0, 'corpus', label)
    out.to_csv(os.path.join(TABLES, f'TABLE_2a3_CorpusVsBothArms_{label}.csv'),
               index=False)
    per = pd.concat(detail)
    arms = list(out.empirical_arm)
    wide = per.pivot(index='metric', columns='empirical_arm',
                     values='w1_standardized')
    wide['change'] = wide[arms[1]] - wide[arms[0]]
    wide = wide.sort_values(arms[1], ascending=False)
    wide.to_csv(os.path.join(TABLES, f'TABLE_2a3_PerCharacteristic_{label}.csv'))

    pd.set_option('display.width', 200)
    print(out.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))
    print()
    print(wide.to_string(float_format=lambda v: f'{v:8.4f}'))

    a, b = out.iloc[0], out.iloc[1]
    do = abs(b.weighted_objective - a.weighted_objective)
    dm = abs(b.mode_tv - a.mode_tv)
    dv = abs(b.visible_tv - a.visible_tv)
    print('\n=== the criterion ===')
    print(f'  objective moved   {do:.4f}   noise sd {NOISE_OBJECTIVE_SD:.4f}   '
          f'{do / NOISE_OBJECTIVE_SD:.2f} sd')
    print(f'  Silverman mode TV moved {dm:.4f}   noise {NOISE_MODE_TV:.4f}')
    print(f'  visible   mode TV moved {dv:.4f}   noise {NOISE_MODE_TV:.4f}')
    inside = (do < NOISE_OBJECTIVE_SD and dm < NOISE_MODE_TV
              and dv < NOISE_MODE_TV)
    print(f'  inside seed-to-seed noise on all three: {"YES" if inside else "NO"}')
    print(f'  -> {"do NOT regenerate" if inside else "RETUNE and regenerate once"}')
    print('\n  READ THIS BEFORE ACTING ON THE LINE ABOVE. The quantity measured\n'
          '  is how far the objective moves when the REFERENCE changes, and part\n'
          '  of it is an irreducible difference between two reference sets rather\n'
          '  than a mismatch a corpus can close. It is the right input to the\n'
          '  decision ONCE, for the corpus that predates the split. For a corpus\n'
          '  generated AFTER it, the question is whether the objective against\n'
          '  the split arm improved, which the four-way table answers.')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '2026-09-13b')

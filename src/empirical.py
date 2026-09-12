"""Prepare the empirical EC3 ECC datasets for analysis.

The source is a frozen, dated raw extract under `data/raw/`: one row per EPD,
carrying the embodied carbon coefficient and enough provenance to audit it. It
is raw in the sense that matters here, that no outlier rule has ever been
applied to it, which is what lets the cleaning rule below treat both ends of the
distribution the same way.

Three choices are applied here, and each moves numbers.

1. Cleaning is multiplicative and symmetric. An ECC is strictly positive and
   right skewed, so an additive interquartile rule is the wrong shape: the low
   bound `Q1 - 3 * IQR` is negative in most categories and therefore never
   binds, which trims high outliers while leaving values many orders of
   magnitude below the mean in place. The rule is applied in log space instead,
   at both ends. See `datageneration.clean_empirical_symmetric`.

2. Point weights are drawn from a flat Dirichlet, alpha = 1, matching the
   synthetic arm. Market shares are unknown, and a flat Dirichlet is the
   maximum-entropy prior over them.

   A note on direction, because it is easy to state backwards. alpha is the
   Dirichlet CONCENTRATION parameter; a smaller alpha gives MORE dispersed
   market shares. It is still a lower bound on reality: at n = 100 a flat
   Dirichlet gives an expected largest share of 5.2 percent, while Marsh, Hattam
   and Allen (2025) report Rest-of-World BOF steel at 63.75 percent of global
   production.

3. Each dataset is divided by its own UNWEIGHTED mean, matching the synthetic
   path exactly. See decision 6 in CLAUDE.md.

A category is kept only if at least `min_n` values survive cleaning. That
threshold, not the extraction, is what decides how many categories the empirical
arm holds, so the count is reported rather than assumed.
"""

import json
import os

import numpy as np
import pandas as pd

from datageneration import clean_empirical_symmetric

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED = os.path.join(ROOT, 'data', 'processed')
RAW = os.path.join(ROOT, 'data', 'raw')

#: The frozen raw extract the analysis reads. A dated file, never overwritten.
SOURCE = os.path.join(RAW, 'ec3_raw_ecc_2026-08-14.csv.gz')

DIRICHLET_ALPHA = 1.0
CLEAN_IQR_MULT = 3.0
MIN_N = 3


def load_raw(path=SOURCE):
    """The raw ECC values per material category, uncleaned."""
    df = pd.read_csv(path, usecols=['material_query', 'ecc'], low_memory=False)
    return {mat: g.ecc.to_numpy(float)
            for mat, g in df.groupby('material_query', sort=True)}


def source_meta(path=SOURCE):
    with open(path.replace('.csv.gz', '_runmeta.json')) as f:
        return json.load(f)


def prepare(rng, path=SOURCE, alpha=DIRICHLET_ALPHA, mult=CLEAN_IQR_MULT,
            min_n=MIN_N):
    """Clean, weight and normalize. Returns (datasets, report).

    Pass a DEDICATED generator, `rng.spawn(1)[0]`, not the notebook's shared
    one. The weights drawn here would otherwise depend on how many random
    values had been consumed before this function was called, so adding a
    figure earlier in the notebook would silently change every empirical
    metric. A spawned generator depends only on how many times `spawn` has been
    called, which makes the empirical datasets reproducible independently of
    what else the notebook does.

    `datasets[mat]` is (values, weights) with values divided by their
    unweighted mean.
    """
    raw = load_raw(path)
    out, report = {}, []
    for mat in sorted(raw):
        data = raw[mat]
        kept = clean_empirical_symmetric(data, mult)
        removed = len(data) - len(kept)
        row = dict(material=mat, n_before=len(data), n_after=len(kept),
                   n_removed=removed,
                   n_removed_low=int((data < kept.min()).sum()) if len(kept) else 0,
                   n_removed_high=int((data > kept.max()).sum()) if len(kept) else 0,
                   min_over_mean_before=float(data.min() / data.mean()),
                   max_over_mean_before=float(data.max() / data.mean()),
                   min_over_mean_after=(float(kept.min() / kept.mean())
                                        if len(kept) else np.nan),
                   max_over_mean_after=(float(kept.max() / kept.mean())
                                        if len(kept) else np.nan))
        if len(kept) < min_n:
            row['status'] = 'dropped_too_few_values'
            report.append(row)
            continue
        row['status'] = 'ok'
        report.append(row)
        w = rng.dirichlet(np.ones(len(kept)) * alpha)
        out[mat] = (kept / np.mean(kept), w)
    return out, report


def write(datasets, report, label, out_root=PROCESSED, meta_extra=None):
    """Write a prepared empirical arm to a new, dated file. Never overwrites."""
    path = os.path.join(out_root, f'empirical_{label}.json')
    if os.path.exists(path):
        raise FileExistsError(f'{path} already exists; pick a new label.')
    payload = {mat: {'data': v.tolist(), 'weights': w.tolist()}
               for mat, (v, w) in datasets.items()}
    with open(path, 'w') as f:
        json.dump(payload, f)
    meta = dict(
        label=label, source=os.path.relpath(SOURCE, ROOT),
        source_meta=source_meta(),
        dirichlet_alpha=DIRICHLET_ALPHA, clean_iqr_mult=CLEAN_IQR_MULT,
        clean_rule='multiplicative log-space IQR, both ends',
        min_n=MIN_N,
        n_materials=len(datasets),
        n_values_before=int(sum(r['n_before'] for r in report)),
        n_values_after=int(sum(r['n_after'] for r in report)),
        n_values_removed=int(sum(r['n_removed'] for r in report)),
        n_materials_dropped=int(sum(r['status'] != 'ok' for r in report)),
        report=report,
    )
    if meta_extra:
        meta.update(meta_extra)
    with open(os.path.join(out_root, f'empirical_{label}_runmeta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return path, meta


def load(label, out_root=PROCESSED):
    with open(os.path.join(out_root, f'empirical_{label}.json')) as f:
        d = json.load(f)
    return {mat: (np.asarray(v['data'], float), np.asarray(v['weights'], float))
            for mat, v in d.items()}

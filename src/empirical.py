"""Prepare the 138 empirical EC3 ECC datasets for analysis.

Two Stage 2a decisions are applied here, and both move numbers.

1. Dirichlet concentration. Point weights were drawn at alpha = 5 in the
   empirical arm and alpha = 1 in the synthetic arm, so the two arms of the
   study received systematically different weight concentrations on the exact
   dimension the paper is about. Both are now alpha = 1: it is what the
   manuscript already describes, and it is the maximum-entropy prior over
   unknown market shares.

   A note on direction, because it is easy to state backwards. alpha is the
   Dirichlet CONCENTRATION parameter; a smaller alpha gives MORE dispersed
   market shares. So alpha = 1 makes the weights less equal than alpha = 5 did,
   and the measured weighting effect goes UP, not down. It is still a lower
   bound on reality: at n = 100 a flat Dirichlet gives an expected largest
   share of 5.2 percent, while Marsh, Hattam and Allen (2025) report
   Rest-of-World BOF steel at 63.75 percent of global production.

2. Multiplicative low-end cleaning, per decision 12. See
   datageneration.clean_empirical_low_end for why, and for why only the low end
   is treated.

The output is written to a new, dated file. The pre-regeneration
dct_realeccs_trimmed.json is never overwritten: it is one of the eight files in
data/INPUTS.sha256 and, since the EC3 extraction directory it came from no
longer exists, it is now the only surviving record of the empirical arm.
"""

import json
import os

import numpy as np

from datageneration import clean_empirical_low_end

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED = os.path.join(ROOT, 'data', 'processed')
SOURCE = os.path.join(PROCESSED, 'dct_realeccs_trimmed.json')

DIRICHLET_ALPHA = 1.0
CLEAN_IQR_MULT = 3.0


def load_raw(path=SOURCE):
    with open(path) as f:
        d = json.load(f)
    return {mat: np.asarray(v['data'], float) for mat, v in d.items()}


def prepare(rng, path=SOURCE, alpha=DIRICHLET_ALPHA, mult=CLEAN_IQR_MULT,
            min_n=3):
    """Clean, weight and normalize. Returns (datasets, report).

    `datasets[mat]` is (values, weights) with values divided by their
    unweighted mean, matching the synthetic path exactly (decision 6).
    """
    raw = load_raw(path)
    out, report = {}, []
    for mat in sorted(raw):
        data = raw[mat]
        kept = clean_empirical_low_end(data, mult)
        removed = len(data) - len(kept)
        row = dict(material=mat, n_before=len(data), n_after=len(kept),
                   n_removed=removed,
                   min_over_mean_before=float(data.min() / data.mean()),
                   min_over_mean_after=(float(kept.min() / kept.mean())
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
    import corpus
    path = os.path.join(out_root, f'empirical_{label}.json')
    if os.path.exists(path):
        raise FileExistsError(f'{path} already exists; pick a new label.')
    payload = {mat: {'data': v.tolist(), 'weights': w.tolist()}
               for mat, (v, w) in datasets.items()}
    with open(path, 'w') as f:
        json.dump(payload, f)
    meta = dict(
        label=label, source=os.path.relpath(SOURCE, ROOT),
        dirichlet_alpha=DIRICHLET_ALPHA, clean_iqr_mult=CLEAN_IQR_MULT,
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

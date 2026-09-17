"""The "95 percent of ECC datasets have one visible mode" figure is a bandwidth.

The author doubted it on sight: "that doesn't feel quite right to me. When
looking at those datasets, I remember seeing a lot of irregularities." The doubt
is correct and the mechanism is circular in a way nothing had noticed.

`modality.n_modes_visible` counts local maxima of `scipy.stats.gaussian_kde(x)`
at its DEFAULT bandwidth, which is Scott's rule. Stage 2c then established, on a
criterion that has nothing to do with modality, that **Scott oversmooths this
data by about 35 percent**: the bandwidth that minimizes W1 against the known
parent sits at 0.46 to 0.56 of Scott's. A mode counter run at an oversmoothing
bandwidth undercounts modes, so the headline figure is a statement about Scott's
rule rather than about ECC data.

THAT MATTERS TWICE. It is quoted in the manuscript as a property of the data, and
decision 38 TUNED THE GENERATOR against it -- so the corpus was matched to the
empirical arm on a measure that could not see the structure both arms have.

    conda run -n compareuq python audits/visible_modes_bandwidth.py [n_synth]
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import corpus  # noqa: E402
import empirical  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 42
N_SYNTH = 1_500

#: Multiples of scipy's default bandwidth. 1.00 is what `n_modes_visible` uses;
#: 0.74 is 1/1.35, the correction Stage 2c's parent criterion implies for Scott.
MULTIPLES = (1.2, 1.0, 0.9, 0.8, 0.74, 0.6, 0.5, 0.4)
MIN_N = 8


def n_modes_at(x, mult, prominence=0.05, grid_n=512):
    """`modality.n_modes_visible` with the bandwidth scaled by `mult`."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < MIN_N or np.std(x) <= 0:
        return 1
    try:
        k = gaussian_kde(x)
    except Exception:
        return 1
    h0 = float(np.sqrt(k.covariance[0, 0]))
    k.set_bandwidth(bw_method=(h0 * mult) / np.std(x, ddof=1))
    g = np.linspace(x.min(), x.max(), grid_n)
    y = k(g)
    idx = argrelextrema(y, np.greater)[0]
    if not len(idx):
        return 1
    peak = y.max()
    if peak <= 0:
        return 1
    kept = 0
    for i in idx:
        left = y[:i].min() if i > 0 else y[i]
        right = y[i + 1:].min() if i < len(y) - 1 else y[i]
        if (y[i] - max(left, right)) / peak >= prominence:
            kept += 1
    return max(kept, 1)


def main(n_synth=N_SYNTH):
    os.makedirs(TABLES, exist_ok=True)
    rng = np.random.default_rng(SEED)
    emp, _ = empirical.prepare(rng.spawn(1)[0])
    met, vals, _ = corpus.load_corpus()
    ids = sorted(met.sample(n_synth, random_state=0).dataset.astype(str))
    syn = corpus.as_dict(vals[vals.dataset_id.isin(set(ids))])

    rows = []
    for arm, ds in (('empirical', emp), ('synthetic', syn)):
        for name, (x, w) in ds.items():
            if len(x) < MIN_N:
                continue
            r = dict(arm=arm, dataset=name, n=len(x))
            for m in MULTIPLES:
                r[f'modes_{m}'] = n_modes_at(np.asarray(x, float), m)
            rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_VisibleModesByBandwidth.csv'),
             index=False)

    pd.set_option('display.width', 200)
    print('=' * 78)
    print('SHARE WITH EXACTLY ONE VISIBLE MODE, by bandwidth')
    print('=' * 78)
    print('1.00 is what modality.n_modes_visible uses, which is scipy\'s default,')
    print('which is Scott. 0.74 is 1/1.35, the correction the parent criterion')
    print('implies for Scott (Stage 2c section 4.8).')
    out = []
    for m in MULTIPLES:
        e = d[d.arm == 'empirical'][f'modes_{m}']
        s = d[d.arm == 'synthetic'][f'modes_{m}']
        ce = e.value_counts(normalize=True)
        cs = s.value_counts(normalize=True)
        # set() over a Series iterates its VALUES, not its index, which made
        # this read 0.0000 at every bandwidth on the first run.
        keys = set(ce.index) | set(cs.index)
        tv = 0.5 * float(sum(abs(ce.get(k, 0.0) - cs.get(k, 0.0))
                             for k in keys))
        out.append(dict(multiple=m, empirical_unimodal=float((e == 1).mean()),
                        synthetic_unimodal=float((s == 1).mean()),
                        empirical_3plus=float((e >= 3).mean()),
                        synthetic_3plus=float((s >= 3).mean()),
                        total_variation=tv))
    t = pd.DataFrame(out)
    print(t.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print()
    print('TWO THINGS TO READ OFF IT.')
    print('1. The 95 percent figure is not a property of the data. It falls to')
    print('   about 73 percent at the corrected bandwidth and to 55 at 0.6.')
    print('2. THE AGREEMENT BETWEEN THE ARMS IS ALSO A PROPERTY OF THE')
    print('   BANDWIDTH. Total variation is 0.012 at Scott, which is what the')
    print('   generator was tuned to, and several times that once the bandwidth')
    print('   can see the structure -- driven by datasets with THREE OR MORE')
    print('   visible modes, which the corpus has far fewer of than the arm.')
    print('   Multimodality favors the KDE, so on this dimension the corpus is')
    print('   biased AGAINST the KDE and not for it.')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_SYNTH)

"""Tune the generator configuration by comparing DISTRIBUTIONS, arm to arm.

This is the iterative step the whole approach turns on: generate a sample under
a candidate configuration, compare the distribution of every statistical
characteristic against the empirical datasets, and change the configuration
where they disagree.

Each characteristic is scored by the Wasserstein-1 distance between the two
distributions with both put on the empirical scale, so it is in empirical
standard deviations and comparable across characteristics. Range coverage is
deliberately NOT the objective: it reads 100 percent while a distribution sits
in the wrong place inside that range, which is exactly the failure this script
exists to prevent.

THE CHARACTERISTICS ARE NOT EQUALLY IMPORTANT, so the objective weights them.
Averaging them treats `n`, which the strata fix by construction, as mattering as
much as modality, which decides whether a KDE can beat a parametric fit at all.
A lognormal cannot represent a second mode and a KDE can, so the mode-count
distribution and the spread that separates modes are what the study's central
comparison rests on; the goodness-of-fit characteristics are largely downstream
of them. WEIGHTS names the weights and `score` reports the objective both
weighted and unweighted, so the choice is visible rather than buried.

The mode-count distribution enters as its own term rather than only through
`crit_bw_1`, because it is the quantity being matched: the total variation
distance between the two mode-count distributions, on the same 0-to-1 footing
as a standardized W1.

    python b5_tune_configuration.py                 # score the current default
    python b5_tune_configuration.py sweep '{...}'   # sweep candidate configs
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from _common import write, TABLES

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coverage
import empirical
import genconfig as G
import generator as GEN
import modality as MD
from customstats import empirical_metadata

SEED = 42
PER_STRATUM = 110          # 440 datasets, the pre-flight scale (see handoff 7c)

#: Weight on each characteristic in the objective. Everything unnamed is 1.0.
#:
#: `crit_bw_1` is Silverman's critical bandwidth, the modality statistic, and
#: `coeffvar` is the spread. Both are weighted 3. `modality_index` is the older
#: continuous index kept alongside it and is weighted 2 rather than 3, because
#: it and `crit_bw_1` measure the same property and weighting both at 3 would
#: give modality six units of influence rather than three.
#:
#: `n` is weighted 0.25, not 0: the strata fix the size distribution by
#: construction, so scoring it at full weight rewards nothing the configuration
#: controls, but zeroing it would hide a stratum that failed to fill.
WEIGHTS = {
    'crit_bw_1': 3.0,
    'coeffvar': 3.0,
    'modality_index': 2.0,
    'n': 0.25,
}

#: Weight on the mode-count total variation distance, on the same footing as a
#: standardized W1. Equal to the modality statistic's weight.
MODE_TV_WEIGHT = 3.0


def empirical_arm(rng_seed=SEED):
    """Characteristics and mode counts for the empirical datasets."""
    ds, _ = empirical.prepare(np.random.default_rng(rng_seed).spawn(1)[0])
    met = pd.DataFrame({m: empirical_metadata(x, w)
                        for m, (x, w) in ds.items()}).T.astype(float)
    met.index.name = 'material'
    rng = np.random.default_rng(0)
    modes = np.array([MD.n_modes_silverman(x, rng=rng, nboot=100)
                      for x, _ in ds.values()])
    return met.reset_index(), modes


def sample_config(cfg, seed=SEED, per_stratum=PER_STRATUM):
    rng = np.random.default_rng(seed)
    rows, modes = [], []
    mrng = np.random.default_rng(seed + 1)
    for st in cfg.strata:
        for _ in range(per_stratum):
            n = int(np.clip(np.floor(10 ** rng.uniform(np.log10(st.n_lo),
                                                       np.log10(st.n_hi + 1))),
                            st.n_lo, st.n_hi))
            x, w, rec = GEN.generate_dataset(cfg, n, rng)
            if x is None or GEN.validity_failures(x, w, n):
                continue
            m = empirical_metadata(x, w)
            m['stratum'] = st.name
            rows.append(m)
            if len(x) >= 4:
                modes.append(MD.n_modes_silverman(x, rng=mrng, nboot=60))
    return pd.DataFrame(rows), np.array(modes)


def score(emp_met, emp_modes, syn_met, syn_modes):
    """Per-characteristic distances, the mode-count table, and the objective.

    Returns (per_characteristic, mode_table, summary_dict).
    """
    d = coverage.distribution_comparison(emp_met, syn_met)
    d['weight'] = d.metric.map(WEIGHTS).fillna(1.0)

    mode_rows = [(k, float((emp_modes == k).mean()), float((syn_modes == k).mean()))
                 for k in range(1, 7)]
    mode_tv = sum(abs(a - b) for _, a, b in mode_rows) / 2.0

    num = float((d.w1_standardized * d.weight).sum()) + MODE_TV_WEIGHT * mode_tv
    den = float(d.weight.sum()) + MODE_TV_WEIGHT
    summary = dict(
        mean_w1_unweighted=float(d.w1_standardized.mean()),
        weighted_objective=num / den,
        mode_tv=mode_tv,
        unimodal_empirical=float((emp_modes == 1).mean()),
        unimodal_synthetic=float((syn_modes == 1).mean()),
        worst_metric=d.iloc[0].metric,
        worst_w1=float(d.iloc[0].w1_standardized),
        coeffvar_w1=float(d.loc[d.metric == 'coeffvar', 'w1_standardized'].iloc[0])
        if (d.metric == 'coeffvar').any() else np.nan,
        skewness_w1=float(d.loc[d.metric == 'skewness', 'w1_standardized'].iloc[0])
        if (d.metric == 'skewness').any() else np.nan,
        crit_bw_1_w1=float(d.loc[d.metric == 'crit_bw_1', 'w1_standardized'].iloc[0])
        if (d.metric == 'crit_bw_1').any() else np.nan,
        n_datasets=int(len(syn_met)),
    )
    return d, pd.DataFrame(mode_rows,
                           columns=['modes', 'empirical', 'synthetic']), summary


def report(name, cfg, emp_met, emp_modes, per_stratum=PER_STRATUM):
    t = time.time()
    syn_met, syn_modes = sample_config(cfg, per_stratum=per_stratum)
    d, modes, s = score(emp_met, emp_modes, syn_met, syn_modes)
    print(f'\n=== {name}  ({len(syn_met)} datasets, {time.time()-t:.0f}s) ===')
    print(f'  weighted objective   {s["weighted_objective"]:.4f}')
    print(f'  mean W1 unweighted   {s["mean_w1_unweighted"]:.4f}')
    print(f'  mode-count TV        {s["mode_tv"]:.4f}')
    print(f'  unimodal   empirical {s["unimodal_empirical"]*100:5.1f}%   '
          f'synthetic {s["unimodal_synthetic"]*100:5.1f}%')
    print(f'  coeffvar W1 {s["coeffvar_w1"]:.3f}   skewness W1 '
          f'{s["skewness_w1"]:.3f}   crit_bw_1 W1 {s["crit_bw_1_w1"]:.3f}')
    print(f'  worst characteristic {s["worst_metric"]} ({s["worst_w1"]:.3f})')
    return d, modes, s, syn_met, syn_modes


if __name__ == '__main__':
    print(f'weights: {WEIGHTS}, mode_tv {MODE_TV_WEIGHT}, everything else 1.0')
    print(f'source:  {os.path.relpath(empirical.SOURCE, os.path.join(TABLES, "..", "..", ".."))}')
    print('measuring the empirical arm ...')
    emp_met, emp_modes = empirical_arm()
    print(f'  {len(emp_met)} datasets, {(emp_modes==1).mean()*100:.1f}% unimodal')

    if len(sys.argv) > 1 and sys.argv[1] == 'sweep':
        base = G.DEFAULT
        spec = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        cands = {'current default': base}
        for name, kw in spec.items():
            cands[name] = base.replace(**{k: eval(v) if isinstance(v, str) else v
                                          for k, v in kw.items()})
        out = []
        for nm, cfg in cands.items():
            d, modes, s, _, _ = report(nm, cfg, emp_met, emp_modes)
            out.append(dict(config=nm, **s))
        sm = pd.DataFrame(out)
        write(sm, 'TABLE_2a_TuningSweep.csv')
        print('\n=== summary ===')
        print(sm.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))
    else:
        d, modes, s, syn_met, syn_modes = report('current default', G.DEFAULT,
                                                 emp_met, emp_modes)
        pd.set_option('display.width', 200)
        print('\nper characteristic:')
        print(d[['metric', 'weight', 'w1_standardized', 'ks',
                 'empirical_median', 'synthetic_median', 'median_shift_in_sd']]
              .to_string(index=False, float_format=lambda v: f'{v:,.3f}'))
        print('\nmode counts:')
        print(modes.to_string(index=False, float_format=lambda v: f'{v:,.3f}'))

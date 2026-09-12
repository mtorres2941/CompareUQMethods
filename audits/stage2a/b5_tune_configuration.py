"""Tune the generator configuration by comparing DISTRIBUTIONS, arm to arm.

This is the iterative step the whole approach turns on: generate a sample under
a candidate configuration, compare the distribution of every statistical
characteristic against the 138 empirical datasets, and change the configuration
where they disagree.

The score is the Wasserstein-1 distance between the two distributions with both
put on the empirical scale, so it is in empirical standard deviations and
comparable across characteristics. Range coverage is deliberately NOT the
objective: it reads 100 percent while a distribution sits in the wrong place
inside that range, which is exactly the failure this script exists to prevent.

Modality is scored separately and reported in full, because the number of modes
is a count and its distribution is the thing most easily got wrong.

    python b5_tune_configuration.py            # score the current default
    python b5_tune_configuration.py sweep      # sweep the overlap range
"""
import os, sys, time, warnings
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
PER_STRATUM = 110


def empirical_arm():
    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0])
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
    d = coverage.distribution_comparison(emp_met, syn_met)
    mode_rows = []
    for k in range(1, 7):
        mode_rows.append((k, float((emp_modes == k).mean()),
                          float((syn_modes == k).mean())))
    mode_err = sum(abs(a - b) for _, a, b in mode_rows)     # total variation x2
    return d, pd.DataFrame(mode_rows,
                           columns=['modes', 'empirical', 'synthetic']), mode_err


def report(name, cfg, emp_met, emp_modes):
    t = time.time()
    syn_met, syn_modes = sample_config(cfg)
    d, modes, mode_err = score(emp_met, emp_modes, syn_met, syn_modes)
    print(f'\n=== {name}  ({len(syn_met)} datasets, {time.time()-t:.0f}s) ===')
    print(f'mean W1 across characteristics : {d.w1_standardized.mean():.3f}')
    print(f'worst characteristic           : {d.iloc[0].metric} '
          f'({d.iloc[0].w1_standardized:.3f})')
    print(f'unimodal share  empirical {(emp_modes==1).mean()*100:5.1f}%   '
          f'synthetic {(syn_modes==1).mean()*100:5.1f}%')
    print(f'mode-count total variation     : {mode_err/2:.3f}')
    return d, modes, mode_err, syn_met, syn_modes


if __name__ == '__main__':
    print('measuring the empirical arm ...')
    emp_met, emp_modes = empirical_arm()
    print(f'  138 datasets, {(emp_modes==1).mean()*100:.1f}% unimodal')

    if len(sys.argv) > 1 and sys.argv[1] == 'sweep':
        base = G.DEFAULT
        import json as _json
        spec = _json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        hi_ov = dict(overlap_log10_lo=np.log10(0.3), overlap_log10_hi=np.log10(1.4))
        cands = {'overlap 0.3-1.4 (round 1 winner)': base.replace(**hi_ov)}
        for name, kw in spec.items():
            cands[name] = base.replace(**hi_ov, **{k: eval(v) if isinstance(v, str) else v
                                                   for k, v in kw.items()})
        out = []
        for nm, cfg in cands.items():
            d, modes, err, _, _ = report(nm, cfg, emp_met, emp_modes)
            out.append(dict(config=nm, mean_w1=d.w1_standardized.mean(),
                            worst=d.iloc[0].metric,
                            worst_w1=d.iloc[0].w1_standardized,
                            mode_tv=err / 2))
        s = pd.DataFrame(out)
        write(s, 'TABLE_2a_TuningSweep.csv')
        print('\n=== summary ===')
        print(s.to_string(index=False, float_format=lambda v: f'{v:,.3f}'))
    else:
        d, modes, err, syn_met, syn_modes = report('current default', G.DEFAULT,
                                                   emp_met, emp_modes)
        pd.set_option('display.width', 200)
        print('\nper characteristic:')
        print(d.to_string(index=False, float_format=lambda v: f'{v:,.3f}'))
        print('\nmode counts:')
        print(modes.to_string(index=False, float_format=lambda v: f'{v:,.3f}'))

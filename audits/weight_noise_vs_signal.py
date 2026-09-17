"""Is the variable-weighted penalty the WEIGHTS, or the flat Dirichlet stand-in?

The author's objection to scoring against the known parent: "variable data is
parent distribution + noise, so it's not a faithful representation of the parent.
If we compare to the parent, we'd have to just look at the uniform data."

THE OBJECTION IS HALF RIGHT AND THE HALF THAT IS RIGHT MATTERS. The synthetic
market-weighted parent IS a real population object -- market share attaches at
the MODE level with `mode_coupling = 1.0`, so a variable-weighted method is
estimating something that exists. But the weights a dataset actually carries are
built in two steps: mode k is given its true market share, and that share is then
split among the points inside mode k by a FLAT DIRICHLET. The first step is
signal. The second is pure noise, and it is noise the real world does not have,
because a real market share is a property of a product, not a random draw.

So the measured penalty for variable weighting may be a property of the STAND-IN
rather than of weighting as a practice, and that distinction decides how the
paper's weighting claim should be worded.

WHAT THIS MEASURES. The same variable-weighted fits under two weight vectors:

  `realized`  the weights the corpus actually stores. Mode-level market share,
              split within each mode by a flat Dirichlet. This is what every
              other table in the study uses.
  `oracle`    the same mode-level market share, split EQUALLY within each mode.
              Same signal, no within-mode noise. Not available to a practitioner
              and not proposed as a method; it is the counterfactual that
              isolates the noise.

If the penalty vanishes under `oracle`, it is the stand-in. If it survives, it is
the weighting.

This needs the MODE ASSIGNMENT of every point, which the corpus does not store,
so it replays the generator exactly as `corpus.rebuild_parents` does.

    conda run -n compareuq python audits/weight_noise_vs_signal.py [n_keep]
"""
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import corpus  # noqa: E402
import fitting as FT  # noqa: E402
import genconfig as G  # noqa: E402
import generator as GEN  # noqa: E402
import recovery as R  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
N_KEEP = 2_000


def oracle_weights(parent, modes):
    """Mode-level market share, split equally inside each mode.

    The same total weight per mode that `generator.draw_weights` targets, with
    the within-mode flat Dirichlet replaced by equal shares. A mode that drew no
    points contributes nothing and the rest are renormalized, which is what the
    realized weights do too.
    """
    modes = np.asarray(modes, int)
    k = len(parent.comps)
    counts = np.bincount(modes, minlength=k).astype(float)
    # `generator.draw_weights` targets `parent.market`, which equals
    # `market_effective` at the configured coupling of 1.0. Mirror the
    # source rather than the identity, so this stays right if coupling moves.
    share = np.asarray(parent.market, float).copy()
    share[counts == 0] = 0.0
    total = share.sum()
    if not total > 0:
        return np.ones(len(modes)) / len(modes)
    share = share / total
    per_point = np.zeros(k)
    nz = counts > 0
    per_point[nz] = share[nz] / counts[nz]
    w = per_point[modes]
    return w / w.sum()


def main(n_keep=N_KEEP):
    os.makedirs(TABLES, exist_ok=True)
    cfg = G.DEFAULT
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    sizes, strata = GEN.stratified_sizes(cfg, rng)
    p = GEN.probe_sizes(cfg, rng)
    sizes = np.concatenate([sizes, p])

    met, _, _ = corpus.load_corpus(with_values=False)
    wanted = set(met.sample(n_keep, random_state=0).dataset.astype(str))

    rows = []
    for i, n in enumerate(sizes):
        ds = f'dataset{i}'
        # The stream must be consumed for EVERY slot, whether kept or not.
        parent, record, x, w, modes = corpus._replay_one(cfg, int(n), rng)
        if parent is None or ds not in wanted:
            continue
        w_oracle = oracle_weights(parent, modes)
        grid = R.recovery_grid(x, w, parent)
        models_real, _ = FT.fit_pewt(x, w)
        models_orac, _ = FT.fit_pewt(x, w_oracle)
        row = dict(dataset=ds, n=len(x), k=len(parent.comps),
                   parent_separation=R.parent_separation(parent, grid))
        for pe in ('Normal', 'Lognormal', 'KDE'):
            row[f'{pe}_uniform'] = R.w1_against_parent(
                models_real[f'{pe}, Uniform'], parent, 'market', grid)
            row[f'{pe}_realized'] = R.w1_against_parent(
                models_real[f'{pe}, Variable'], parent, 'market', grid)
            row[f'{pe}_oracle'] = R.w1_against_parent(
                models_orac[f'{pe}, Variable'], parent, 'market', grid)
        rows.append(row)
        if len(rows) % 250 == 0:
            print(f'  kept {len(rows)}/{n_keep}, slot {i+1}/{len(sizes)}, '
                  f'{time.time()-t0:.0f}s', flush=True)

    d = R.add_size_band(pd.DataFrame(rows))
    d.to_csv(os.path.join(TABLES, 'TABLE_WeightNoiseVsSignal.csv'), index=False)
    report(d)


def report(d):
    pd.set_option('display.width', 220)
    print()
    print('=' * 78)
    print('W1 AGAINST THE MARKET PARENT: uniform, realized weights, oracle weights')
    print('=' * 78)
    for pe in ('Normal', 'Lognormal', 'KDE'):
        t = d.groupby('size_band')[[f'{pe}_uniform', f'{pe}_realized',
                                    f'{pe}_oracle']].mean()
        t.columns = ['uniform', 'realized', 'oracle']
        t['realized_beats_uniform'] = d.groupby('size_band').apply(
            lambda g: float((g[f'{pe}_realized'] < g[f'{pe}_uniform']).mean()),
            include_groups=False)
        t['oracle_beats_uniform'] = d.groupby('size_band').apply(
            lambda g: float((g[f'{pe}_oracle'] < g[f'{pe}_uniform']).mean()),
            include_groups=False)
        print(f'--- {pe} ---')
        print(t.to_string(float_format=lambda v: f'{v:.4f}'))
        print()
    print('=' * 78)
    print('THE VERDICT')
    print('=' * 78)
    print('POSITIVE MEANS VARIABLE WEIGHTING IS BETTER: these are distances, so')
    print('the uniform fit scoring HIGHER is the variable fit winning.')
    print('If `oracle_beats_uniform` is high where `realized_beats_uniform` is')
    print('low, the penalty for variable weighting at small n is the flat')
    print('Dirichlet STAND-IN and not weighting itself, and the paper has to say')
    print('so. If both are low, variable weighting genuinely does not pay at')
    print('small n whatever the weights are.')
    for pe in ('Normal', 'Lognormal', 'KDE'):
        for band in [b[0] for b in R.SIZE_BANDS]:
            g = d[d.size_band == band]
            if len(g) < 20:
                continue
            long = pd.concat([
                g[['dataset']].assign(arm='s', method=m,
                                      w1=g[f'{pe}_{m}'])
                for m in ('uniform', 'realized', 'oracle')])
            b = R.paired_bootstrap(long, 'w1', 'uniform',
                                   rng=np.random.default_rng(0)).set_index(
                'method')
            print(f'  {pe:<10}{band:<14} '
                  f'uniform minus realized {-b.loc["realized","mean_difference"]:+.4f}'
                  f'{"*" if b.loc["realized","distinguishable"] else " "}   '
                  f'uniform minus oracle {-b.loc["oracle","mean_difference"]:+.4f}'
                  f'{"*" if b.loc["oracle","distinguishable"] else " "}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_KEEP)

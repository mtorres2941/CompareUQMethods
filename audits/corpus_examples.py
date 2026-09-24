"""Example datasets per size stratum, for ANY candidate generator config.

WHY THIS EXISTS. Notebook 1 draws this figure for the production configuration
as `CompareUQMethods_SUPP_DatasetExamplesByStratum`, and it is the fastest way
to tell whether a corpus looks like real ECC data. It cannot be pointed at a
CANDIDATE configuration, which is what you need when deciding whether to
regenerate. This draws the same panels from the same `generator.draw_parent`
for any config in `corpus_joint_structure.CANDIDATES`, side by side with real
categories.

IT IS A TUNING INSTRUMENT, NOT A PAPER FIGURE, and it writes under
`outputs/tables/audits/` for that reason. The paper's version stays notebook
1's cell, which is where decision 165 requires figure code to live.

    conda run -n compareuq python audits/corpus_examples.py [candidate ...]
"""
import dataclasses
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, HERE)

import empirical                            # noqa: E402
import genconfig as G                       # noqa: E402
import generator                            # noqa: E402
from corpus_joint_structure import CANDIDATES  # noqa: E402

OUT = os.path.join(ROOT, 'outputs', 'tables', 'audits')
NSHOW = 12


def panel(ax, x, title, parent=None, color='tab:blue'):
    """One dataset: its parent density where one exists, its values as ticks.

    Lifted from notebook 1's cell so the two figures are the same picture.
    """
    x = np.asarray(x, float)
    if parent is not None:
        scale = parent.normalizer
        g = np.linspace(parent.lo / scale, parent.hi / scale, 600)
        ax.fill_between(g, parent.pdf(g), color=color, alpha=0.35, linewidth=0)
        ax.set_xlim(0, max(np.percentile(x, 99.5) * 1.15, x.max() * 0.35))
    elif len(x) >= 2 and np.std(x) > 0:
        kde = gaussian_kde(x)
        g = np.linspace(max(0.0, x.min() - 0.1 * np.ptp(x)),
                        x.max() + 0.1 * np.ptp(x), 400)
        ax.fill_between(g, kde(g), color=color, alpha=0.35, linewidth=0)
        ax.set_xlim(0, np.percentile(x, 99.5) * 1.15)
    ax.scatter(x, [0] * len(x), marker='|', s=60, color='black',
               alpha=min(1.0, 30 / max(len(x), 1)) if len(x) > 30 else 0.9,
               linewidth=0.6)
    ax.set_ylim(0,)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.tick_params(labelsize=5)
    ax.set_title(title, fontsize=5.5)


def draw(name, overrides, empirical_pick, seed=20260923):
    cfg = dataclasses.replace(G.DEFAULT, **overrides) if overrides else G.DEFAULT
    rng = np.random.default_rng(seed)
    nrows = len(cfg.strata) + 1
    fig, axes = plt.subplots(nrows, NSHOW, figsize=(18, 1.55 * nrows),
                             gridspec_kw=dict(hspace=1.0, wspace=0.12))
    for r, stratum in enumerate(cfg.strata):
        for c in range(NSHOW):
            n = int(np.clip(10 ** rng.uniform(np.log10(stratum.n_lo),
                                              np.log10(stratum.n_hi + 1)),
                            stratum.n_lo, stratum.n_hi))
            parent = None
            for _ in range(cfg.max_parent_retries):
                parent, _rec = generator.draw_parent(cfg, n, rng)
                if parent is not None:
                    break
            if parent is None:
                axes[r, c].set_axis_off()
                continue
            x, _modes = parent.sample(n, rng)
            x = x / np.mean(x)
            panel(axes[r, c], x,
                  f'n={n}  CV={np.std(x) / np.mean(x):.2f}\n'
                  f'{len(parent.comps)} component(s)', parent=parent)
            if c == 0:
                axes[r, c].set_ylabel(stratum.name, fontsize=7)
                axes[r, c].get_yaxis().set_visible(True)
                axes[r, c].set_yticks([])
    for c, (mat, x) in enumerate(empirical_pick):
        panel(axes[-1, c], x,
              f'{mat[:16]}\nn={len(x)}  CV={np.std(x) / np.mean(x):.2f}',
              color='tab:orange')
        if c == 0:
            axes[-1, c].set_ylabel('REAL', fontsize=7)
            axes[-1, c].get_yaxis().set_visible(True)
            axes[-1, c].set_yticks([])
    extra = ', '.join(f'{k}={v}' for k, v in overrides.items()) or 'as shipped'
    fig.suptitle(f'Candidate "{name}"  [{extra}]\n'
                 'blue: the exact parent each synthetic dataset was drawn '
                 'from.  orange: a KDE of real values, the only density real '
                 'data has.', fontsize=9)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'FIG_CorpusExamples_{name}.png')
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    return path


def main(argv):
    wanted = argv[1:] or list(CANDIDATES)
    # The real arm is BUILT, not stored: `empirical.prepare` is what notebook 1
    # calls, and it applies the category rules and the cleaning. Rebuilding it
    # here rather than reading a stale json is what keeps this figure's bottom
    # row the same 147 categories the study uses.
    dct, _report = empirical.prepare(np.random.default_rng(0))
    dct = {k: np.asarray(v[0], float) / np.mean(v[0])
           for k, v in dct.items() if len(v[0]) >= 3}
    names = sorted(dct, key=lambda m: len(dct[m]))
    pick = [(names[int(round(q * (len(names) - 1)))],
             dct[names[int(round(q * (len(names) - 1)))]])
            for q in np.linspace(0.05, 0.95, NSHOW)]
    for name in wanted:
        if name not in CANDIDATES:
            raise SystemExit(f'unknown candidate {name!r}; '
                             f'have {sorted(CANDIDATES)}')
        print(draw(name, CANDIDATES[name], pick), flush=True)


if __name__ == '__main__':
    main(sys.argv)

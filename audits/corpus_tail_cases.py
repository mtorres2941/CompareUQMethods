"""Do the corpus's over-represented cases move any headline? Stage 2h.

Two of this project's open items are of the same shape: the corpus holds more
of some kind of dataset than the real categories do, and the question that
decides whether to act is not how big the mismatch is but whether anything
depends on it. Decision 138 answered one that way -- the dispersion shortfall
is one contaminated EC3 bin and excising it moves every headline by less than
half a percentage point -- and this applies the same test to two more.

    SIX OR MORE MODES. About 5 to 6 percent of the corpus against 0.7 percent
    of the real categories, on record since Stage 2a-2 (decision 37). The
    mechanism is the same one decision 168 names: the generator reaches a mode
    COUNT by separating components, which is a different shape from a shoulder
    on a skewed body.

    MULTIMODAL AND DISPERSED AT ONCE. 1.9 percent of the corpus against 16.2
    percent of the real categories, which is the joint-structure gap decisions
    168 to 170 measured and declined to engineer away.

WHAT THIS DOES, in both cases: excise the cell from the corpus and re-measure
the headline figures on what is left. A headline that does not move is a
mismatch that bounds the study's COVERAGE and not its conclusions, which is a
limitation to state rather than a corpus to rebuild.

Generation is closed (decisions 47, 48 and 55) and nothing here reopens it.

    conda run -n compareuq python audits/corpus_tail_cases.py
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import corpus                     # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
OUT = os.path.join(ROOT, 'outputs', 'tables')

#: What counts as widely dispersed. The median real category sits at about
#: 0.63, so this is the upper part of the real range rather than a cut chosen
#: to make a number come out.
DISPERSED_CV = 1.0


def headlines(scores, label):
    """The method comparison's headline figures on a subset of the corpus.

    `scores` is one row per (dataset, method) carrying the score against the
    market-weighted true parent, which is the only target under which the two
    weighting schemes are comparable (decision 65).
    """
    piv = scores.pivot_table(index=['dataset', 'n'], columns='method',
                            values='w1_market').dropna().reset_index()
    methods = [c for c in piv.columns if c not in ('dataset', 'n')]
    best = piv[methods].idxmin(axis=1)
    row = dict(subset=label, n_datasets=len(piv))
    for m in methods:
        row[f'closest_{m}'] = float((best == m).mean())
    k, lg = 'KDE, Uniform', 'Lognormal, Uniform'
    if k in piv and lg in piv:
        wins = (piv[k] < piv[lg]).astype(float)
        row['kde_beats_lognormal'] = float(wins.mean())
        # The size at which the kernel estimate overtakes, by a logistic on
        # log(n), which is the same inversion the rest of the project uses.
        import flip as FL
        beta = FL.logistic_fit(piv.n.to_numpy(float), wins.to_numpy(float))
        row['crossover_n'] = FL.logistic_crossing(beta, 0.5)
    for m in methods:
        row[f'mean_w1_{m}'] = float(piv[m].mean())
    return row


def main():
    os.makedirs(TABLES, exist_ok=True)
    pd.set_option('display.width', 240)
    met, _, _ = corpus.load_corpus(with_values=False)
    met = met[met.dataset.astype(str).str.startswith('dataset')].copy()

    path = os.path.join(OUT, 'TABLE_TargetComparison.csv')
    if not os.path.exists(path):
        print(f'{path} is missing; run notebook 2 first.')
        return
    scores = pd.read_csv(path)
    scores = scores[scores.arm == 'synthetic'].copy()
    scores['dataset'] = scores.dataset.astype(str)
    met['dataset'] = met.dataset.astype(str)

    # THE MODE COUNT IS NOT IN THE CORPUS'S CHARACTERISTICS and has to be read
    # from the table notebook 1 writes. It is counted at the bandwidth the
    # study FITS, which is the density a reader is shown and the pLCA samples
    # from (decision 82), not at scipy's default, which oversmooths this data
    # by about 35 percent and undercounts modes for that reason.
    vm_path = os.path.join(OUT, 'TABLE_VisibleModes.csv')
    vm = pd.read_csv(vm_path)
    vm = vm[vm.arm == 'synthetic'].copy()
    vm['dataset'] = vm.dataset.astype(str)
    met = met.merge(vm[['dataset', 'modes_fitted', 'modes_scipy_default']],
                    on='dataset', how='left')
    met['many_modes'] = (met.modes_fitted >= 6) | (met.modes_scipy_default >= 6)
    met['dispersed'] = met.coeffvar >= DISPERSED_CV
    # THE JOINT CELL USES DECISION 168'S OWN DEFINITION, which is TWO OR MORE
    # visible modes and not six. Six-or-more was a separate concern, raised in
    # Stage 2a-2 from a Silverman critical-bandwidth count on a corpus that has
    # since been regenerated twice; measured at the bandwidth the study fits,
    # the current corpus has no dataset with six visible modes at all.
    met['multimodal'] = met.modes_fitted >= 2
    met['both'] = met.multimodal & met.dispersed
    print(f'mode counts available for {int(met.modes_fitted.notna().sum())} '
          f'of {len(met)} datasets')

    print('WHAT SHARE OF THE CORPUS EACH CELL IS')
    for c in ('many_modes', 'multimodal', 'dispersed', 'both'):
        print(f'  {c:<12s} {met[c].mean():.4f}  ({int(met[c].sum())} datasets)')
    print()

    keys = met.set_index('dataset')
    rows = [headlines(scores, 'whole corpus')]
    print('CONDITIONAL STRUCTURE, which is what decision 168 measures:')
    disp = met[met.dispersed]
    print(f'  P(multimodal | dispersed) on the corpus = '
          f'{disp.multimodal.mean():.4f} over {len(disp)} datasets')
    print()
    for c, label in (('many_modes', 'excluding 6+ modes'),
                     ('multimodal', 'excluding 2+ modes'),
                     ('dispersed', 'excluding CV >= 1.0'),
                     ('both', 'excluding multimodal AND dispersed')):
        drop = set(keys.index[keys[c]])
        rows.append(headlines(scores[~scores.dataset.isin(drop)], label))
        rows.append(headlines(scores[scores.dataset.isin(drop)],
                              f'ONLY {label.split("excluding ")[1]}'))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_CorpusTailCases.csv'), index=False)
    cols = ['subset', 'n_datasets', 'kde_beats_lognormal', 'crossover_n']
    cols += [c for c in d.columns if c.startswith('closest_')]
    print('DOES EXCLUDING THE CELL MOVE A HEADLINE?')
    print(d[[c for c in cols if c in d]]
          .to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print()
    print('A headline that does not move when the cell is excised is a')
    print('COVERAGE limitation to state, not a corpus to rebuild. A cell whose')
    print('OWN row differs sharply from the whole corpus is one the study')
    print('cannot speak to, which is the thing to say in the limitations.')


if __name__ == '__main__':
    main()

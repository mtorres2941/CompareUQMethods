"""Stage 2a, Part 0 item 1c: the effective size of the shipped corpus.

The paper's headline percentages, and every interval quoted across datasets,
assume the 10,000 datasets are independent draws. Under the collapse they are
not: datasets sharing a component type and count share one underlying vector of
standard normal (or t, or skew-normal) draws, and every later step is monotone,
so the value ORDER is preserved. Two datasets with |Spearman rho| = 1 are the
same random draw wearing different location, scale and power parameters.

This script clusters the corpus on that relation and counts connected
components, exactly rather than by sampling, to give an effective corpus size.

Writes TABLE_2a_CollapseEffectiveN.csv
"""
import numpy as np, pandas as pd, sys, os
from scipy.stats import rankdata
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from _common import load_shipped, write

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RHO_CUT = 0.999
NMIN = 10


def cluster_arm(DATA, keys, label):
    bysize = {}
    for k in keys:
        d = np.asarray(DATA[k]['data'], float)
        bysize.setdefault(len(d), []).append(k)

    total = len(keys)
    small = sum(len(v) for n, v in bysize.items() if n < NMIN)
    edges_i, edges_j = [], []
    idx_of = {k: i for i, k in enumerate(keys)}
    tested = 0
    for n, ks in bysize.items():
        if n < NMIN or len(ks) < 2:
            continue
        R = np.array([rankdata(np.asarray(DATA[k]['data'], float)) for k in ks])
        Z = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
        C = np.abs(Z @ Z.T) / n
        iu = np.triu_indices(len(ks), 1)
        tested += len(iu[0])
        hit = C[iu] > RHO_CUT
        for a, b in zip(iu[0][hit], iu[1][hit]):
            edges_i.append(idx_of[ks[a]]); edges_j.append(idx_of[ks[b]])

    g = coo_matrix((np.ones(len(edges_i)), (edges_i, edges_j)), shape=(total, total))
    ncomp, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    # only components containing an n >= NMIN dataset are meaningfully clustered
    n_in_multi = int(np.sum(sizes[sizes > 1]))
    return dict(arm=label, n_datasets=total, n_excluded_small_n=small,
                pairs_tested=tested, duplicate_edges=len(edges_i),
                duplicate_pair_rate=len(edges_i) / tested if tested else np.nan,
                connected_components=int(ncomp),
                datasets_in_a_duplicate_cluster=n_in_multi,
                frac_in_a_duplicate_cluster=n_in_multi / total,
                largest_cluster=int(sizes.max()),
                effective_corpus_size=int(ncomp),
                redundancy_pct=100 * (1 - ncomp / total))


if __name__ == '__main__':
    DATA_ship, keep, _, _ = load_shipped()
    rows = [cluster_arm(DATA_ship, keep, 'shipped_analysed_10k'),
            cluster_arm(DATA_ship, list(DATA_ship), 'shipped_all_15k')]

    import legacy_generator_pre_stage1 as legacy   # noqa
    from a1_collapse_corpus_damage import gen_legacy, gen_current
    print('regenerating both arms for the same measurement ...')
    DL = gen_legacy(); rows.append(cluster_arm(DL, list(DL), 'legacy_rerun_15k'))
    DC = gen_current(); rows.append(cluster_arm(DC, list(DC), 'current_15k'))

    df = pd.DataFrame(rows)
    write(df, 'TABLE_2a_CollapseEffectiveN.csv')
    pd.set_option('display.width', 250)
    print(df.to_string(index=False))

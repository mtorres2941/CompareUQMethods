"""Generate, write and read a synthetic ECC corpus.

This module is what retires `generate_dontread`, the module-level boolean in
notebook 1 that decided whether to regenerate or read from disk. A hand-edited
flag is the wrong mechanism for the one irreversible operation in the project:
it leaves no record in the outputs of which mode produced them, so a corpus on
disk could not be told apart from any other.

In its place, a corpus is a NAMED, DATED DIRECTORY that carries its own
provenance. Regeneration is an explicit act -- running
`python -m corpus <label>` or calling `generate_corpus` -- and the notebooks
only ever read. `data/processed/CORPUS.json` names the active corpus, so
repointing the analysis is a one-line change to a tracked file rather than an
edit to a notebook cell.

Layout of `data/processed/corpus_<label>/`:

    values.parquet    long format: dataset_id, value, weight. Decision 15:
                      columnar, compresses well, readable from R and Julia,
                      which matters for a Zenodo deposit
    metrics.parquet   one row per dataset: stratum, the statistical metrics,
                      and the validity-filter verdict
    parents.json.gz   the provenance record per dataset, enough to rebuild the
                      parent CDF exactly. This is what Stage 2c scores against
    runmeta.json      seed, full configuration, git commit, library versions,
                      platform, timings, and the counts of everything the
                      generator had to retry or reject

Nothing here overwrites anything. `generate_corpus` refuses to write into a
directory that already exists.
"""

import gzip
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import genconfig as G
import generator as GEN
from customstats import empirical_metadata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED = os.path.join(ROOT, 'data', 'processed')
POINTER = os.path.join(PROCESSED, 'CORPUS.json')


# --------------------------------------------------------------------------
def _git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'unknown'


def _git_dirty():
    try:
        out = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT,
                                      text=True, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return None


def run_metadata(cfg, extra=None):
    import scipy
    meta = dict(
        created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        seed=cfg.seed,
        config=cfg.to_dict(),
        git_commit=_git_commit(),
        git_working_tree_dirty=_git_dirty(),
        python=sys.version.split()[0],
        numpy=np.__version__,
        scipy=scipy.__version__,
        pandas=pd.__version__,
        platform=platform.platform(),
        machine=platform.machine(),
    )
    if extra:
        meta.update(extra)
    return meta


# --------------------------------------------------------------------------
def generate_corpus(cfg, label, out_root=PROCESSED, include_probe=True,
                    progress=True):
    """Generate a corpus and write it, with its provenance, to a new directory.

    Returns the directory path. Refuses to overwrite.
    """
    out = os.path.join(out_root, f'corpus_{label}')
    if os.path.exists(out):
        raise FileExistsError(
            f'{out} already exists. A corpus is never overwritten: pick a new '
            f'label. The pre-regeneration baseline lives in '
            f'data/baseline_frozen/ and data/INPUTS.sha256.')
    os.makedirs(out)

    rng = np.random.default_rng(cfg.seed)
    sizes, strata = GEN.stratified_sizes(cfg, rng)
    if include_probe:
        p = GEN.probe_sizes(cfg, rng)
        sizes = np.concatenate([sizes, p])
        strata = np.concatenate([strata, np.array([cfg.probe.name] * len(p))])

    t0 = time.time()
    ids, vals, wts = [], [], []
    metric_rows, records = [], {}
    n_failed_parent = 0
    failed_parents = {}
    invalid = {}
    retries_total = 0

    for i, (n, st) in enumerate(zip(sizes, strata)):
        ds = f'dataset{i}'
        x, w, rec = GEN.generate_dataset(cfg, int(n), rng)
        if x is None:
            # Record WHY, not just that it happened. A count alone leaves the
            # next reader with a corpus that is short by one and no way to find
            # out what went wrong with it.
            n_failed_parent += 1
            failed_parents[ds] = dict(n=int(n), stratum=str(st),
                                      status=rec.get('status'),
                                      statuses=rec.get('statuses'))
            continue
        fails = GEN.validity_failures(x, w, int(n))
        if fails:
            invalid[ds] = fails
            continue
        retries_total += rec.get('component_retries', 0)
        m = empirical_metadata(x, w)
        m['dataset'] = ds
        m['stratum'] = st
        m['is_probe'] = st == cfg.probe.name
        m['k'] = rec['k']
        m['overlap_target'] = rec['overlap_target']
        m['overlap_achieved'] = rec['overlap_achieved']
        m['overlap_status'] = rec['overlap_status']
        m['truncated_mass'] = rec['truncated_mass']
        m['normalizer'] = rec['normalizer']
        m['n_components_dropped'] = rec['n_components_dropped']
        metric_rows.append(m)
        records[ds] = rec
        ids.append(np.full(len(x), ds))
        vals.append(x)
        wts.append(w)
        if progress and (i + 1) % 1000 == 0:
            print(f'  {i+1:>6} / {len(sizes)}   {time.time()-t0:6.0f}s', flush=True)

    values = pd.DataFrame({
        'dataset_id': pd.Categorical(np.concatenate(ids)),
        'value': np.concatenate(vals).astype('float64'),
        'weight': np.concatenate(wts).astype('float64'),
    })
    values.to_parquet(os.path.join(out, 'values.parquet'), index=False,
                      compression='zstd')

    metrics = pd.DataFrame(metric_rows)
    cols = ['dataset', 'stratum', 'is_probe'] + [c for c in metrics.columns
                                                 if c not in ('dataset', 'stratum',
                                                              'is_probe')]
    metrics = metrics[cols]
    metrics.to_parquet(os.path.join(out, 'metrics.parquet'), index=False,
                       compression='zstd')

    with gzip.open(os.path.join(out, 'parents.json.gz'), 'wt') as f:
        json.dump(records, f)

    meta = run_metadata(cfg, dict(
        label=label,
        n_requested=int(len(sizes)),
        n_written=int(len(metrics)),
        n_corpus=int((~metrics.is_probe).sum()),
        n_probe=int(metrics.is_probe.sum()),
        include_probe=include_probe,
        n_failed_parent=n_failed_parent,
        failed_parents=failed_parents,
        n_invalid=len(invalid),
        invalid_reasons=_count_reasons(invalid),
        component_retries_total=int(retries_total),
        overlap_status_counts=metrics.overlap_status.value_counts().to_dict(),
        wall_seconds=round(time.time() - t0, 1),
        files={fn: os.path.getsize(os.path.join(out, fn))
               for fn in sorted(os.listdir(out))},
    ))
    with open(os.path.join(out, 'runmeta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out, 'invalid_datasets.json'), 'w') as f:
        json.dump(invalid, f, indent=2)
    return out


def _count_reasons(invalid):
    from collections import Counter
    c = Counter()
    for reasons in invalid.values():
        for r in reasons:
            c[r] += 1
    return dict(c)


# --------------------------------------------------------------------------
def set_active(label, pointer=POINTER):
    with open(pointer, 'w') as f:
        json.dump({'active_corpus': f'corpus_{label}',
                   'note': 'Written by src/corpus.py. The notebooks read this '
                           'to find the corpus; they never regenerate.'}, f, indent=2)


def active_dir(pointer=POINTER, out_root=PROCESSED):
    with open(pointer) as f:
        return os.path.join(out_root, json.load(f)['active_corpus'])


def load_corpus(directory=None, with_values=True, corpus_only=True):
    """Read a corpus. Returns (metrics, values_or_None, runmeta)."""
    d = directory or active_dir()
    metrics = pd.read_parquet(os.path.join(d, 'metrics.parquet'))
    if corpus_only:
        metrics = metrics[~metrics.is_probe].reset_index(drop=True)
    with open(os.path.join(d, 'runmeta.json')) as f:
        meta = json.load(f)
    values = None
    if with_values:
        values = pd.read_parquet(os.path.join(d, 'values.parquet'))
        if corpus_only:
            values = values[values.dataset_id.isin(set(metrics.dataset))]
    return metrics, values, meta


def load_parents(directory=None):
    d = directory or active_dir()
    with gzip.open(os.path.join(d, 'parents.json.gz'), 'rt') as f:
        return json.load(f)


def as_dict(values):
    """Long-format values -> {dataset_id: (data, weights)}, for code that still
    wants the old shape."""
    out = {}
    for ds, g in values.groupby('dataset_id', observed=True):
        out[str(ds)] = (g['value'].to_numpy(), g['weight'].to_numpy())
    return out


# --------------------------------------------------------------------------
def scaled_config(cfg, n_total, n_probe=None):
    """The same configuration at a smaller corpus size, for iteration.

    Every stratum is scaled by the same factor, so the design is unchanged and
    only the precision of each estimate drops. A draft corpus is for deciding
    whether the GENERATOR is right; the full corpus is for the paper's numbers.
    Generating 10,000 datasets to answer a question that 1,000 answers is about
    twelve wasted minutes per iteration, plus three to five times the cost of
    every downstream check.

    The label must record it. `runmeta.json` carries n_corpus, so a draft cannot
    be mistaken for the real thing after the fact, but the directory name is
    what a reader sees first.
    """
    k = len(cfg.strata)
    per = max(1, int(round(n_total / k)))
    strata = tuple(G.Stratum(st.name, st.n_lo, st.n_hi, per) for st in cfg.strata)
    probe = cfg.probe
    if n_probe is not None:
        probe = G.Stratum(probe.name, probe.n_lo, probe.n_hi, int(n_probe))
    return cfg.replace(strata=strata, probe=probe)


if __name__ == '__main__':
    label = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y%m%d')
    cfg = G.DEFAULT
    if len(sys.argv) > 2:
        n_total = int(sys.argv[2])
        cfg = scaled_config(cfg, n_total, n_probe=max(5, n_total // 200))
        print(f'DRAFT SIZE: {cfg.n_datasets} datasets, '
              f'{cfg.probe.n_datasets} probe. Not a corpus for the paper.')
    print(f'generating corpus {label} with seed {cfg.seed} ...')
    d = generate_corpus(cfg, label)
    with open(os.path.join(d, 'runmeta.json')) as f:
        meta = json.load(f)
    print(json.dumps({k: v for k, v in meta.items() if k != 'config'}, indent=2))
    print(f'\nwrote {d}')
    print('run  python -c "import sys; sys.path.insert(0,\'src\'); '
          f'import corpus; corpus.set_active(\'{label}\')"  to activate it')


# --------------------------------------------------------------------------
# adapters, so downstream notebook cells keep the shape they already use
# --------------------------------------------------------------------------
METRIC_COLUMNS = ('n', 'mean', 'mean_uw', 'coeffvar', 'coeffvar_uw', 'skewness',
                  'skewness_uw', 'kurtosis', 'kurtosis_uw', 'entropy', 'entropy_uw',
                  'modality_index', 'modality_index_uw', 'crit_bw_1', 'crit_bw_1_uw',
                  'weight_outliers', 'weight_outliers_uw', 'fit_norm_SW',
                  'fit_norm_SW_uw', 'fit_lognorm_SW', 'fit_lognorm_SW_uw',
                  'w_v_uw_wasserstein')


def as_legacy_dict(metrics, values):
    """{dataset_id: {'data', 'weights', 'metrics'}}, the shape notebooks 2 and 3
    already consume. The corpus itself is stored in long format (decision 15);
    this only reshapes it in memory."""
    cols = [c for c in METRIC_COLUMNS if c in metrics.columns]
    meta = metrics.set_index('dataset')[cols].to_dict('index')
    out = {}
    for ds, g in values.groupby('dataset_id', observed=True):
        ds = str(ds)
        if ds not in meta:
            continue
        out[ds] = {'data': g['value'].to_numpy(),
                   'weights': g['weight'].to_numpy(),
                   'metrics': meta[ds]}
    return out


def make_combos(metrics, rng, nmats=4, corpus_only=True):
    """Disjoint groups of `nmats` datasets, for the pLCA.

    Written into the corpus directory rather than to a loose combos.txt, so a
    grouping always belongs to a named corpus. The probe set is excluded: its
    only job is to test whether results plateau above n = 10 ** 4, and it must
    stay out of every aggregate.

    THE REMAINDER IS HELD OUT, AND IT IS NAMED. `corpus_2026-09-14d` holds
    9,999 datasets rather than 10,000, because one parent failed to solve and
    was reported rather than approximated (decision 22), so the corpus no longer
    divides by four. Every group is exactly `nmats` datasets and the leftover
    `len(ids) % nmats` are excluded from the pLCA entirely.

    A short last group is the alternative and it is wrong here. Every downstream
    rank metric -- `eci_rank_1` through `eci_rank_4`, and the two reduction-rank
    families -- is a rank among exactly four materials, and the headline result
    is a FREQUENCY over those ranks. One group of three would leave
    `eci_rank_4` undefined for that group and would make a rank-1 frequency of
    1/3 comparable with one of 1/4 in the same aggregate. Dropping the remainder
    costs three datasets of 9,999; keeping it would put a different estimand in
    the same column.

    `combos_holdout` names which datasets were held out. Do not read the
    remainder off an integer division somewhere downstream: report it.
    """
    m = metrics[~metrics.is_probe] if (corpus_only and 'is_probe' in metrics) else metrics
    ids = np.array(sorted(m['dataset'].astype(str)))
    rng.shuffle(ids)
    ids = ids[:len(ids) // nmats * nmats]
    return ids.reshape(-1, nmats)


def combos_holdout(metrics, combos, corpus_only=True):
    """The corpus datasets that no pLCA group contains, sorted.

    See `make_combos` for why the remainder is held out rather than forming a
    short last group.
    """
    m = metrics[~metrics.is_probe] if (corpus_only and 'is_probe' in metrics) else metrics
    ids = set(m['dataset'].astype(str))
    return sorted(ids - set(np.asarray(combos).astype(str).ravel()))


def describe_combos(metrics, combos, corpus_only=True):
    """One line stating the grouping, so the remainder cannot fall out silently."""
    held = combos_holdout(metrics, combos, corpus_only)
    n_groups, nmats = np.asarray(combos).shape
    m = metrics[~metrics.is_probe] if (corpus_only and 'is_probe' in metrics) else metrics
    line = (f'{n_groups:,} pLCA groups, every one of {nmats} datasets, '
            f'covering {n_groups * nmats:,} of {len(m):,} corpus datasets')
    if held:
        line += (f'\nheld out of the pLCA grouping ({len(held)}, the remainder '
                 f'of {len(m):,} / {nmats}): ' + ', '.join(held))
    else:
        line += '\nno dataset is excluded'
    return line


def write_combos(directory, combos):
    path = os.path.join(directory, 'combos.csv')
    pd.DataFrame(combos, columns=[f'material{i+1}' for i in range(combos.shape[1])]
                 ).to_csv(path, index=False)
    return path


def load_combos(directory=None):
    d = directory or active_dir()
    return pd.read_csv(os.path.join(d, 'combos.csv')).to_numpy(dtype=str)

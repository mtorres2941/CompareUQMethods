"""Shared helpers for the an earlier revision audit scripts."""
import json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
FROZEN = os.path.join(ROOT, 'data', 'baseline_frozen', 'data', 'processed')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.makedirs(TABLES, exist_ok=True)


def load_shipped(frozen=True, with_values=True):
    """Load the shipped 15,000 synthetic datasets and the analysed subset."""
    base = FROZEN if frozen else os.path.join(ROOT, 'data', 'processed')
    with open(os.path.join(base, 'DATA_all.json')) as f:
        DATA_all = json.load(f)
    with open(os.path.join(base, 'datasets_outliers.json')) as f:
        outliers = set(json.load(f))
    with open(os.path.join(base, 'datasets_trimto10k.json')) as f:
        trim = set(json.load(f))
    keep = [k for k in DATA_all if k not in outliers and k not in trim]
    if not with_values:
        for k in DATA_all:
            DATA_all[k].pop('data', None)
            DATA_all[k].pop('weights', None)
    return DATA_all, keep, outliers, trim


def metrics_frame(DATA):
    """Metrics dict-of-dicts -> DataFrame, index = dataset id."""
    return pd.DataFrame({k: v['metrics'] for k, v in DATA.items()}).T


def load_empirical(frozen=True):
    base = FROZEN if frozen else os.path.join(ROOT, 'data', 'processed')
    with open(os.path.join(base, 'dct_realeccs_trimmed.json')) as f:
        d = json.load(f)
    for mat in d:
        d[mat]['data'] = np.array(d[mat]['data'], dtype=float)
        d[mat]['weights'] = np.array(d[mat]['weights'], dtype=float)
    return d


def write(df, name, note=None):
    path = os.path.join(TABLES, name)
    df.to_csv(path, index=False)
    print(f'wrote {os.path.relpath(path, ROOT)}  ({len(df)} rows)')
    return path

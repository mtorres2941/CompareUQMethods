"""Tests for corpus.remetric_corpus's handling of the derived replay caches."""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import corpus  # noqa: E402


def _tiny_corpus(root, label='src', n_datasets=4):
    """A corpus directory with just enough in it to be remetriced."""
    d = root / f'corpus_{label}'
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, recs = [], []
    for i in range(n_datasets):
        x = rng.lognormal(0.0, 0.4, 12)
        x = x / x.mean()
        w = np.full(12, 1 / 12)
        rows.append(pd.DataFrame(dict(dataset_id=f'dataset{i}', value=x,
                                      weight=w)))
        recs.append(dict(dataset=f'dataset{i}', stratum='s2_10_99',
                         is_probe=False, k=1, overlap_target=0.5,
                         overlap_achieved=0.5, overlap_status='ok',
                         truncated_mass=0.0, normalizer=1.0,
                         n_components_dropped=0))
    values = pd.concat(rows, ignore_index=True)
    values['dataset_id'] = values.dataset_id.astype('category')
    values.to_parquet(d / 'values.parquet', index=False)
    metrics = pd.DataFrame(recs)
    metrics['coeffvar'] = 0.4
    metrics.to_parquet(d / 'metrics.parquet', index=False)
    (d / 'runmeta.json').write_text(json.dumps(dict(label=label, seed=42)))
    (d / 'invalid_datasets.json').write_text('{}')
    with gzip.open(d / corpus.PARENT_SPEC_FILE, 'wt') as f:
        json.dump(dict(corpus=f'corpus_{label}', seed=42, values_verified=True,
                       n_datasets=n_datasets,
                       specs={f'dataset{i}': {'stub': i}
                              for i in range(n_datasets)}), f)
    return d


def test_remetric_relabels_the_parent_spec_cache_it_copies(tmp_path):
    """The cache names the corpus it was built for and `load_parent_specs`
    refuses a mismatch. That guard is right and must stay -- a cache from a
    genuinely different corpus must not be read -- but a copy made by
    `remetric_corpus` is not from a different corpus in any sense that
    matters, because values.parquet is copied byte for byte.

    Notebook 2 failed on exactly this, nineteen minutes into a run, the first
    time a remetriced corpus was used.
    """
    _tiny_corpus(tmp_path)
    out = Path(corpus.remetric_corpus('dst', source=str(tmp_path / 'corpus_src'),
                                      out_root=str(tmp_path), progress=False))
    specs = corpus.load_parent_specs(str(out))
    assert len(specs) == 4

    with gzip.open(out / corpus.PARENT_SPEC_FILE, 'rt') as f:
        payload = json.load(f)
    assert payload['corpus'] == 'corpus_dst'
    # the provenance is not lost by the relabelling
    assert payload['copied_from'] == 'corpus_src'


def test_a_cache_from_a_genuinely_different_corpus_is_still_refused(tmp_path):
    """The guard the relabelling must not weaken."""
    _tiny_corpus(tmp_path)
    out = Path(corpus.remetric_corpus('dst', source=str(tmp_path / 'corpus_src'),
                                      out_root=str(tmp_path), progress=False))
    with gzip.open(out / corpus.PARENT_SPEC_FILE, 'rt') as f:
        payload = json.load(f)
    payload['corpus'] = 'corpus_somewhere_else'
    with gzip.open(out / corpus.PARENT_SPEC_FILE, 'wt') as f:
        json.dump(payload, f)
    with pytest.raises(ValueError, match='was built for'):
        corpus.load_parent_specs(str(out))


def test_remetric_copies_the_values_byte_for_byte(tmp_path):
    import hashlib

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    src = _tiny_corpus(tmp_path)
    out = Path(corpus.remetric_corpus('dst', source=str(src),
                                      out_root=str(tmp_path), progress=False))
    assert sha(src / 'values.parquet') == sha(out / 'values.parquet')
    # and the characteristics were actually recomputed
    new = pd.read_parquet(out / 'metrics.parquet')
    assert 'fit_norm_SF' in new.columns
    assert new.coeffvar.iloc[0] != pytest.approx(0.4)

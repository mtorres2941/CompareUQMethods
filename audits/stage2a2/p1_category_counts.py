"""How many EPDs EC3 holds today in each of the 138 categories.

One cheap request per category, reading x-total-count. Run before the pull so
its size is known in advance and so every category has an independently
obtained expected count to verify the pull against. Counting rows is not that
check; see ../EPDsFromEC3/PULLING_EPDS.md section 1.

Counts are taken twice, with and without the validity filter, because the
2026-03 pull that produced dct_realeccs_trimmed.json used the ec3 library's
default, which silently drops expired declarations.

    conda run -n lucid_lca python p1_category_counts.py
"""
import datetime as dt
import json
import os
import sys

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
EC3DIR = os.path.abspath(os.path.join(ROOT, '..', 'EPDsFromEC3'))
OUT = os.path.join(ROOT, 'outputs', 'tables', 'stage2a2')
API = 'https://buildingtransparency.org/api/epds'


def token():
    for line in open(os.path.join(EC3DIR, '.env')):
        if line.startswith('EC3_KEY='):
            return line.split('=', 1)[1].strip()
    raise RuntimeError('EC3_KEY not found in ../EPDsFromEC3/.env')


def categories():
    """The 138 category names in the empirical arm, with their EC3 ids."""
    with open(os.path.join(ROOT, 'data', 'processed',
                           'dct_realeccs_trimmed.json')) as f:
        mats = sorted(json.load(f))
    with open(os.path.join(EC3DIR, 'data', 'masterformat_ec3_map.json')) as f:
        ids = {e['name']: e['id'] for e in json.load(f)}
    missing = [m for m in mats if m not in ids]
    if missing:
        raise RuntimeError(f'no EC3 id for {missing}')
    return [(m, ids[m]) for m in mats]


def count(session, cid, include_expired, today):
    params = {'category': cid, 'page_size': 1, 'page_number': 1}
    if not include_expired:
        params['date_validity_ends__gt'] = today
    r = session.get(API, params=params, timeout=90)
    r.raise_for_status()
    return int(r.headers.get('x-total-count', -1))


if __name__ == '__main__':
    today = dt.date.today().isoformat()
    s = requests.Session()
    s.headers.update({'Authorization': f'Bearer {token()}'})
    rows = []
    for i, (name, cid) in enumerate(categories(), 1):
        valid = count(s, cid, False, today)
        allrec = count(s, cid, True, today)
        rows.append(dict(material=name, category_id=cid,
                         n_valid=valid, n_including_expired=allrec))
        print(f'{i:3d}/138  {name:<36} valid {valid:>7,}  '
              f'with expired {allrec:>7,}', flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, 'TABLE_2a2_CategoryCounts.csv'), index=False)
    print(f'\ntotal valid            {df.n_valid.sum():,}')
    print(f'total including expired{df.n_including_expired.sum():>9,}')
    print(f'counted on             {today}')

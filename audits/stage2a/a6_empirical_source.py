"""Stage 2a, Part 7: the empirical data at the source.

Three questions:
  1. Were the 138 EC3 datasets deduplicated?
  2. Are industry-average EPDs mixed with product-specific ones in a category?
  3. How much do the empirical metric ranges move under the cleaning rules?

The extraction that produced dct_realeccs_trimmed.json read a directory
'../../EPDsFromEC3/EPD_AllOfEC3' that NO LONGER EXISTS on this machine, and the
file it wrote is post-cleaning, so the shipped empirical arm cannot be
un-trimmed. Questions 1 and 2 are answered from the extraction code plus the
composition of the same 138 EC3 categories in the surviving EPD store; question
3 is answered by re-running the cleaning variants on store-reconstructed
datasets for the same 138 categories.

Writes TABLE_2a_EmpiricalSourceComposition.csv and
       TABLE_2a_EmpiricalCleaningSensitivity.csv
"""
import numpy as np, pandas as pd, sys, os, json, gzip
from _common import load_empirical, write, ROOT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from customstats import empirical_metadata

STORE = os.path.join(ROOT, '..', 'EPDsFromEC3', 'store', 'epd_index.csv.gz')
SEED = 20260911
METRICS = ['n', 'coeffvar', 'entropy', 'skewness', 'kurtosis', 'mode_count_est',
           'weight_outliers', 'fit_norm_SW', 'fit_lognorm_SW', 'w_v_uw_wasserstein']

COLS = ['material_query', 'open_xpd_uuid', 'ec3_internal_id', 'name', 'manufacturer',
        'plant', 'declaration_type', 'manufacturer_specific', 'plant_specific',
        'product_specific', 'gwp_per_declared_unit', 'declared_unit_value',
        'gwp_value', 'gwp_per_kg_value', 'date_validity_ends']


def load_store(mats, latest_pull_only=True):
    """Load the surviving EPD store, restricted to the 138 categories.

    latest_pull_only matters. The store holds up to four pull dates per
    category and re-pulls the same EPDs on each, so counting duplicates across
    the whole store measures the store's own history rather than anything about
    EC3. Restricting to the most recent pull per category is the only honest
    basis for a deduplication claim.
    """
    use = [c for c in COLS if c != 'gwp_per_declared_unit'] + ['pull_date']
    df = pd.read_csv(STORE, usecols=lambda c: c in use, low_memory=False)
    df = df[df.material_query.isin(mats)].copy()
    if latest_pull_only:
        latest = df.groupby('material_query').pull_date.max().rename('_latest')
        df = df.merge(latest, on='material_query')
        df = df[df.pull_date == df._latest].drop(columns='_latest')
    return df


def composition(df):
    rows = []
    for mat, g in df.groupby('material_query'):
        n = len(g)
        dt = g.declaration_type.fillna('missing')
        ind = dt.str.contains('ndustry', na=False) | dt.str.contains('eneric', na=False)
        gw = pd.to_numeric(g.gwp_per_kg_value, errors='coerce')
        ok = gw.notna()
        key = (g.manufacturer.fillna('?').astype(str) + '|'
               + gw.round(6).astype(str))[ok]
        rows.append(dict(
            material=mat, n_records=n,
            n_with_gwp=int(ok.sum()),
            dup_manufacturer_gwp=int(ok.sum() - key.nunique()),
            n_manufacturers=int(g.manufacturer.nunique(dropna=True)),
            top_manufacturer_share=float(g.manufacturer.value_counts(normalize=True).iloc[0])
                if g.manufacturer.notna().any() else np.nan,
            n_unique_uuid=int(g.open_xpd_uuid.nunique(dropna=True)),
            n_missing_uuid=int(g.open_xpd_uuid.isna().sum()),
            dup_uuid_records=int(n - g.open_xpd_uuid.nunique(dropna=True)
                                 - g.open_xpd_uuid.isna().sum()),
            n_unique_name_mfr=int(g.groupby(['name', 'manufacturer'], dropna=False).ngroups),
            frac_industry_wide=float(ind.mean()),
            frac_product_specific=float(g.product_specific.fillna(False).astype(bool).mean()),
            frac_plant_specific=float(g.plant_specific.fillna(False).astype(bool).mean()),
            frac_manufacturer_specific=float(
                g.manufacturer_specific.fillna(False).astype(bool).mean()),
            n_declaration_types=int(dt.nunique()),
            declaration_types='|'.join(sorted(dt.unique())[:6]),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------- cleaning variants --
def clean(data, rule):
    q1, q3 = np.quantile(data, [0.25, 0.75])
    iqr = q3 - q1
    if rule == 'shipped_additive_3iqr':
        return data[(data < q3 + 3 * iqr) & (data > q1 - 3 * iqr)]
    if rule == 'none':
        return data
    if rule == 'additive_1.5iqr':
        return data[(data < q3 + 1.5 * iqr) & (data > q1 - 1.5 * iqr)]
    if rule == 'multiplicative_log_3iqr':
        # symmetric in log space: the low-end bound is a ratio, not a difference,
        # so it binds even when Q1 - 3*IQR is negative.
        L = np.log(data)
        l1, l3 = np.quantile(L, [0.25, 0.75])
        li = l3 - l1
        return data[(L < l3 + 3 * li) & (L > l1 - 3 * li)]
    if rule == 'multiplicative_log_1.5iqr':
        L = np.log(data)
        l1, l3 = np.quantile(L, [0.25, 0.75])
        li = l3 - l1
        return data[(L < l3 + 1.5 * li) & (L > l1 - 1.5 * li)]
    if rule == 'hi_only_3iqr':
        return data[data < q3 + 3 * iqr]
    raise ValueError(rule)


def metrics_under(datasets, rule, seed=SEED):
    rng = np.random.default_rng(seed)
    out = {}
    for mat, data in datasets.items():
        d = clean(np.asarray(data, float), rule)
        if len(d) < 3:
            continue
        w = rng.dirichlet(np.ones_like(d))
        try:
            out[mat] = empirical_metadata(d / np.mean(d), w)
        except Exception:
            continue
    return pd.DataFrame(out).T.astype(float)


if __name__ == '__main__':
    emp = load_empirical()
    mats = sorted(emp)

    print('loading the EPD store ...')
    st = load_store(mats)
    print(f'  {len(st):,} records across {st.material_query.nunique()} of the 138 categories')

    comp = composition(st)
    write(comp, 'TABLE_2a_EmpiricalSourceComposition.csv')
    pd.set_option('display.width', 240)
    print('\n--- Q1 deduplication ---')
    print('measured on the most recent pull per category, so the store\'s own')
    print('re-pull history does not masquerade as duplication in EC3.')
    print(f"records with a duplicate open_xpd_uuid inside their own category : "
          f"{int(comp.dup_uuid_records.sum()):,} of {int(comp.n_records.sum()):,} "
          f"({comp.dup_uuid_records.sum()/comp.n_records.sum()*100:.2f}%)")
    print(f"categories containing at least one such duplicate               : "
          f"{int((comp.dup_uuid_records > 0).sum())} of {len(comp)}")
    print(f"records sharing a (manufacturer, GWP per kg) pair with another record")
    print(f"  in the same category                                         : "
          f"{int(comp.dup_manufacturer_gwp.sum()):,} of {int(comp.n_with_gwp.sum()):,} "
          f"({comp.dup_manufacturer_gwp.sum()/comp.n_with_gwp.sum()*100:.2f}%)")
    print(f"categories containing at least one such pair                  : "
          f"{int((comp.dup_manufacturer_gwp > 0).sum())} of {len(comp)}")
    print(f"median share of a category held by its top manufacturer       : "
          f"{comp.top_manufacturer_share.median()*100:.1f}%  "
          f"(max {comp.top_manufacturer_share.max()*100:.1f}%)")
    dupacross = st.dropna(subset=['open_xpd_uuid']).groupby('open_xpd_uuid').material_query.nunique()
    print(f"EPDs appearing in more than one of the 138 categories           : "
          f"{int((dupacross > 1).sum()):,} of {len(dupacross):,} unique EPDs")
    print(f"records they account for                                       : "
          f"{int(st.open_xpd_uuid.isin(dupacross[dupacross>1].index).sum()):,}")

    print('\n--- Q2 industry-average mixed with product-specific ---')
    print(st.declaration_type.fillna('missing').value_counts().to_string())
    for c in ['manufacturer_specific', 'plant_specific', 'product_specific']:
        if c in st:
            v = st[c].fillna('missing').astype(str).value_counts(normalize=True)
            print(f'  {c:<24} ' + '  '.join(f'{k}={x*100:.1f}%' for k, x in v.items()))
    mixed = comp[(comp.frac_industry_wide > 0) & (comp.frac_industry_wide < 1)]
    print(f"\ncategories mixing industry-wide and other declaration types : "
          f"{len(mixed)} of {len(comp)}")
    print(f"overall industry-wide share                                 : "
          f"{comp.frac_industry_wide.mul(comp.n_records).sum()/comp.n_records.sum()*100:.1f}%")
    print('\nmost mixed categories:')
    print(mixed.nlargest(10, 'frac_industry_wide')[
        ['material', 'n_records', 'frac_industry_wide', 'frac_product_specific']]
        .to_string(index=False, float_format=lambda v: f'{v:,.3f}'))

    print('\n--- Q3 cleaning sensitivity ---')
    # Reconstruct per-category GWP series from the store, mimicking the notebook:
    # gwp per declared unit, non-positive dropped. No dedup, no type filter, so
    # the reconstruction carries the same two choices the original extraction made.
    recon = {}
    for mat, g in st.groupby('material_query'):
        v = pd.to_numeric(g.gwp_per_kg_value, errors='coerce').dropna()
        v = v[v > 0].values
        if len(v) >= 3:
            recon[mat] = v
    print(f'reconstructed {len(recon)} categories from the store')

    rules = ['none', 'shipped_additive_3iqr', 'additive_1.5iqr',
             'multiplicative_log_3iqr', 'multiplicative_log_1.5iqr', 'hi_only_3iqr']
    rows = []
    base = None
    for rule in rules:
        m = metrics_under(recon, rule)
        if rule == 'none':
            base = m
        for col in METRICS:
            s = m[col].replace([np.inf, -np.inf], np.nan).dropna()
            b = base[col].replace([np.inf, -np.inf], np.nan).dropna()
            common = s.index.intersection(b.index)
            rows.append(dict(rule=rule, metric=col, n_datasets=len(m),
                             min=float(s.min()), p05=float(s.quantile(.05)),
                             median=float(s.median()), p95=float(s.quantile(.95)),
                             max=float(s.max()), sd=float(s.std()),
                             mean_abs_shift_vs_none=float(
                                 (s[common] - b[common]).abs().mean()),
                             shift_in_sd_units=float((s[common] - b[common]).abs().mean()
                                                     / b.std()) if b.std() else np.nan))
    sens = pd.DataFrame(rows)
    write(sens, 'TABLE_2a_EmpiricalCleaningSensitivity.csv')
    piv = sens.pivot(index='metric', columns='rule', values='shift_in_sd_units')[
        [r for r in rules if r != 'none']]
    print('\nmean absolute metric shift vs no cleaning, in sd units of the uncleaned metric:')
    print(piv.to_string(float_format=lambda v: f'{v:,.3f}'))
    print('\nfraction of records removed by each rule:')
    for rule in rules:
        kept = sum(len(clean(np.asarray(v, float), rule)) for v in recon.values())
        tot = sum(len(v) for v in recon.values())
        print(f'  {rule:<28} kept {kept:>7,} of {tot:>7,}  removed {100*(1-kept/tot):5.2f}%')

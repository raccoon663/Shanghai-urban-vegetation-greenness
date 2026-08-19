"""Auditable  thermal check using existing  LST (no new egress)."""
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"
TABLES = ROOT / "outputs" / "tables"
v=pd.read_csv(TABLES/"patch_validation.csv"); valid=set(v[v.include_confirmatory.astype(str).str.lower().eq("true")].block_id)
t=pd.read_csv(OUT/"thermal_metrics.csv"); t=t[t.block_id.isin(valid)&(t.n_valid_summer_scenes>=2)].copy()
s=pd.read_csv(TABLES/"validated_primary_sample.csv"); s=s[(s.sensor_strategy=="C_platform_aware")&s.include_confirmatory.astype(str).str.lower().eq("true")]
block_ndvi=s.groupby(["block_id","year"],as_index=False).summer_NDVI.mean(); q=t.merge(block_ndvi,on=["block_id","year"]).dropna()
m1=smf.ols("summer_median_lst_c ~ block_built_fraction + C(year)",q).fit(cov_type="cluster",cov_kwds={"groups":q.block_id})
m2=smf.ols("summer_NDVI ~ summer_median_lst_c + block_built_fraction + C(year)",q).fit(cov_type="cluster",cov_kwds={"groups":q.block_id})
rows=[]
for relation,m,term in [("built_fraction_to_summer_LST",m1,"block_built_fraction"),("summer_LST_to_summer_NDVI_adjusted_built",m2,"summer_median_lst_c")]:
    e=m.params[term]; se=m.bse[term]
    rows.append({"relation":relation,"estimate":e,"standard_error":se,"ci_low":e-1.96*se,"ci_high":e+1.96*se,"p_value":m.pvalues[term],"n_block_years":len(q),"n_blocks":q.block_id.nunique(),"sample_scope":"historically validated original blocks; existing  Landsat LST","replication_thermal_status":"not extracted; new original-block coordinate egress not authorized"})
pd.DataFrame(rows).to_csv(TABLES/"thermal_validation.csv",index=False,encoding="utf-8-sig")
print(pd.DataFrame(rows).to_string(index=False))

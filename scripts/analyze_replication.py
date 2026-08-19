"""Run the fixed standalone replication before any pooled analysis."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; DATA=ROOT/"data"/"processed"
TABLES = ROOT / "outputs" / "tables"
CFG=json.loads((ROOT/"config/final_analysis.json").read_text(encoding="utf-8")); P=CFG["analysis"]
RNG=np.random.default_rng(32026)
val=pd.read_csv(OUT/"replication_patch_validation.csv")
keep=val[val.include_replication_confirmatory.astype(str).str.lower().eq("true")].copy()
ts=pd.read_csv(DATA/"replication_combined_timeseries.csv",parse_dates=["date"])

rows=[]
for p in keep.itertuples(index=False):
    for year in P["years"]:
        g=ts[(ts.patch_id==p.patch_id)&(ts.year==year)].sort_values("date")
        dates=sorted(g.date.dt.date.unique()); active=[d for d in dates if d.month in P["active_months"]]
        gaps=[(b-a).days for a,b in zip(active,active[1:])]
        summer=sum(d.month in P["summer_months"] for d in dates)
        qc=len(dates)>=P["minimum_patch_year_observations"] and summer>=P["minimum_summer_observations"] and bool(gaps) and max(gaps)<=P["maximum_active_season_gap_days"]
        nd=g.ndvi.to_numpy(); sm=g[g.date.dt.month.isin(P["summer_months"])].ndvi.to_numpy()
        rows.append({"sample_origin":"independent_replication","patch_id":p.patch_id,"block_id":p.block_id,"year":year,
          "vegetation_type":p.validated_type,"validation_confidence":p.historical_confidence,
          "built_fraction_1000m":p.built_fraction_1000m,"crop_context_1000m":p.crop_context_1000m,
          "n_observations":len(dates),"summer_observations":summer,"max_active_season_gap_days":max(gaps) if gaps else np.nan,
          "temporal_qc_pass":qc,"include_confirmatory":qc,
          "peak_NDVI":np.nanpercentile(nd,90) if len(nd) else np.nan,"summer_NDVI":np.nanmedian(sm) if len(sm) else np.nan})
sample=pd.DataFrame(rows)
sample.to_csv(OUT/"replication_primary_sample.csv",index=False,encoding="utf-8-sig")

def fit(endpoint):
    q=sample[sample.include_confirmatory].dropna(subset=[endpoint]).copy()
    formula=f"{endpoint} ~ built_fraction_1000m + C(vegetation_type) + C(year) + crop_context_1000m"
    if q.block_id.nunique()<4: raise RuntimeError(f"Insufficient replication blocks for {endpoint}: {q.block_id.nunique()}")
    m=smf.ols(formula,q).fit(cov_type="cluster",cov_kwds={"groups":q.block_id})
    est=m.params["built_fraction_1000m"]; se=m.bse["built_fraction_1000m"]; blocks=q.block_id.unique(); boots=[]; loo=[]
    for _ in range(CFG["replication"]["block_bootstrap_replicates"]):
        parts=[]
        for j,b in enumerate(RNG.choice(blocks,len(blocks),replace=True)):
            z=q[q.block_id==b].copy(); z["boot_block"]=f"r{j}"; parts.append(z)
        try: boots.append(smf.ols(formula,pd.concat(parts,ignore_index=True)).fit().params["built_fraction_1000m"])
        except Exception: pass
    for b in blocks:
        try: loo.append(smf.ols(formula,q[q.block_id!=b]).fit().params["built_fraction_1000m"])
        except Exception: pass
    rho,rp=spearmanr(q.built_fraction_1000m,q[endpoint])
    return {"analysis_stage":"standalone_replication","sensor_strategy":"C_platform_aware","endpoint":endpoint,
      "estimate":est,"standard_error":se,"ci_low":est-1.96*se,"ci_high":est+1.96*se,"p_value":m.pvalues["built_fraction_1000m"],
      "bootstrap_ci_low":np.percentile(boots,2.5),"bootstrap_ci_high":np.percentile(boots,97.5),
      "spearman_rho":rho,"spearman_p":rp,"n_patch_years":len(q),"n_patches":q.patch_id.nunique(),"n_blocks":len(blocks),
      "loo_negative":sum(x<0 for x in loo),"loo_total":len(loo),"loo_min":min(loo),"loo_max":max(loo)}

res=pd.DataFrame([fit("summer_NDVI"),fit("peak_NDVI")])
# This file is the auditable barrier: the combined analysis refuses to run unless it exists.
res.to_csv(TABLES/"replication_results.csv",index=False,encoding="utf-8-sig")
print(res.to_string(index=False))

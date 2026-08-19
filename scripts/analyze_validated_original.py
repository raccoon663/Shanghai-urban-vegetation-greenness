"""Fixed original-sample  validation and three declared sensor strategies."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; DATA=ROOT/"data"/"processed"
TABLES = ROOT / "outputs" / "tables"
CFG=json.loads((ROOT/"config/final_analysis.json").read_text(encoding="utf-8")); P=CFG["analysis"]; RNG=np.random.default_rng(42)
patches=pd.read_csv(OUT/"sampled_patches.csv"); val=pd.read_csv(TABLES/"patch_validation.csv")
lc=pd.read_csv(OUT/"patch_year_landcover_audit.csv")
lc["exclude_probable_conversion"]=lc.exclude_probable_conversion.astype(str).str.lower().eq("true")
s2=pd.read_csv(DATA/"s2_patch_timeseries_harmonized.csv",parse_dates=["date"])
pooled=pd.read_csv(DATA/"combined_patch_timeseries.csv",parse_dates=["date"])
ls=pd.read_csv(DATA/"landsat_patch_timeseries_raw.csv",parse_dates=["date"])
platform=pd.read_csv(TABLES/"platform_calibration.csv")
s2["s2_platform"]=s2.item_id.str.extract(r"^(S2[AB])")
ls["landsat_platform"]=ls.item_id.str.extract(r"^(L[CET]0[789])").replace({"LC08":"Landsat 8","LC09":"Landsat 9","LE07":"Landsat 7"})

# Strategy C: platform-aware coefficients. Landsat 7 is excluded because S2B-L7 fails slope quality;
# all L7 is kept in the matched-pair audit but not used as validated gap fill. L8/L9 use platform-specific
# coefficients; annual instability stays flagged and is not repaired with endpoint-driven year coefficients.
coefs={(r.s2_platform,r.landsat_platform):(r.intercept,r.slope,r.tree_offset) for r in platform.itertuples(index=False)
       if r.landsat_platform in ("Landsat 8","Landsat 9") and .5<=r.slope<=1.5 and r.rmse<=.18}
fills=[]
for lr in ls.itertuples(index=False):
    if lr.landsat_platform not in ("Landsat 8","Landsat 9"): continue
    sg=s2[(s2.patch_id==lr.patch_id)&(s2.year==lr.year)].copy()
    if sg.empty: continue
    # gap-date rule remains >3 days from every S2 date; use the nearest S2 platform to select calibration.
    sg["dd"]=(sg.date-lr.date).abs().dt.days; nearest=sg.sort_values("dd").iloc[0]
    if nearest.dd<=3: continue
    key=(nearest.s2_platform,lr.landsat_platform)
    if key not in coefs: continue
    a,b,t=coefs[key]; nd=np.clip(a+b*lr.ndvi_raw+t*(lr.vegetation_type=="tree"),-.2,1)
    fills.append({"patch_id":lr.patch_id,"block_id":lr.block_id,"vegetation_type":lr.vegetation_type,
                  "date":lr.date,"year":lr.year,"item_id":lr.item_id,"valid_sample_points":lr.valid_sample_points,
                  "ndvi":nd,"source":"Landsat platform-aware validated calibration"})
aware=pd.concat([s2[["patch_id","block_id","vegetation_type","date","year","item_id","valid_sample_points","ndvi","source"]],pd.DataFrame(fills)],ignore_index=True)

def metrics(series,strategy):
    rows=[]
    for p in patches.itertuples(index=False):
        vv=val[val.patch_id==p.patch_id].iloc[0]
        for year in P["years"]:
            g=series[(series.patch_id==p.patch_id)&(series.year==year)].sort_values("date")
            dates=sorted(g.date.dt.date.unique()); active=[d for d in dates if d.month in P["active_months"]]
            gaps=[(b-a).days for a,b in zip(active,active[1:])]; summer=sum(d.month in P["summer_months"] for d in dates)
            temporal=len(dates)>=P["minimum_patch_year_observations"] and summer>=P["minimum_summer_observations"] and bool(gaps) and max(gaps)<=P["maximum_active_season_gap_days"]
            l=lc[(lc.patch_id==p.patch_id)&(lc.year==year)].iloc[0]
            include=bool(vv.include_confirmatory) and temporal and not l.exclude_probable_conversion
            nd=g.ndvi.to_numpy(); sm=g[g.date.dt.month.isin(P["summer_months"])].ndvi.to_numpy()
            rows.append({"sensor_strategy":strategy,"patch_id":p.patch_id,"block_id":p.block_id,"year":year,
                "vegetation_type":vv.validated_type if vv.validated_type in ("tree","grass") else p.vegetation_type,
                "validation_confidence":vv.confidence,"validation_status":vv.validation_status,
                "built_fraction_1000m":p.built_fraction_1000m,"crop_context_1000m":p.crop_context_1000m,
                "n_observations":len(dates),"summer_observations":summer,"max_active_season_gap_days":max(gaps) if gaps else np.nan,
                "temporal_qc_pass":temporal,"include_confirmatory":include,
                "peak_NDVI":np.nanpercentile(nd,90) if len(nd) else np.nan,"summer_NDVI":np.nanmedian(sm) if len(sm) else np.nan})
    return pd.DataFrame(rows)

samples=pd.concat([metrics(s2,"A_S2_only"),metrics(pooled,"B_pooled_calibration"),metrics(aware,"C_platform_aware")],ignore_index=True)
samples.to_csv(TABLES/"validated_primary_sample.csv",index=False,encoding="utf-8-sig")

def fit(d,endpoint):
    q=d[d.include_confirmatory].dropna(subset=[endpoint]).copy()
    formula=f"{endpoint} ~ built_fraction_1000m + C(vegetation_type) + C(year) + crop_context_1000m"
    f=smf.ols(formula,q).fit(cov_type="cluster",cov_kwds={"groups":q.block_id}); est=f.params["built_fraction_1000m"]; se=f.bse["built_fraction_1000m"]
    boots=[]; blocks=q.block_id.unique()
    for _ in range(CFG["replication"]["block_bootstrap_replicates"]):
        draw=RNG.choice(blocks,len(blocks),replace=True); parts=[]
        for j,b in enumerate(draw):
            z=q[q.block_id==b].copy(); z["boot_block"]=f"b{j}"; parts.append(z)
        z=pd.concat(parts,ignore_index=True)
        try: boots.append(smf.ols(formula,z).fit().params["built_fraction_1000m"])
        except Exception: pass
    loo=[]
    for b in blocks:
        z=q[q.block_id!=b]
        try: loo.append(smf.ols(formula,z).fit().params["built_fraction_1000m"])
        except Exception: pass
    rho,rp=spearmanr(q.built_fraction_1000m,q[endpoint])
    return {"endpoint":endpoint,"estimate":est,"standard_error":se,"ci_low":est-1.96*se,"ci_high":est+1.96*se,
        "p_value":f.pvalues["built_fraction_1000m"],"bootstrap_ci_low":np.percentile(boots,2.5),"bootstrap_ci_high":np.percentile(boots,97.5),
        "spearman_rho":rho,"spearman_p":rp,"n_patch_years":len(q),"n_patches":q.patch_id.nunique(),"n_blocks":q.block_id.nunique(),
        "loo_negative":sum(x<0 for x in loo),"loo_total":len(loo),"loo_min":min(loo),"loo_max":max(loo)}

results=[]
for strategy,d in samples.groupby("sensor_strategy"):
    for endpoint in ["summer_NDVI","peak_NDVI"]: results.append({"sensor_strategy":strategy,**fit(d,endpoint)})
pd.DataFrame(results).to_csv(TABLES/"sensor_strategy_results.csv",index=False,encoding="utf-8-sig")
print(pd.DataFrame(results).to_string(index=False))

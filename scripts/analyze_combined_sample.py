"""Compare fixed original/replication results, then fit the combined validated sample."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"
TABLES = ROOT / "outputs" / "tables"
CFG=json.loads((ROOT/"config/final_analysis.json").read_text(encoding="utf-8")); RNG=np.random.default_rng(42026)
rep_path=TABLES/"replication_results.csv"
if not rep_path.exists(): raise RuntimeError("Standalone replication results must be recorded before combination")
origres=pd.read_csv(TABLES/"sensor_strategy_results.csv"); origres=origres[origres.sensor_strategy=="C_platform_aware"].copy()
repres=pd.read_csv(rep_path)
comparison=[]
for endpoint in ["summer_NDVI","peak_NDVI"]:
    o=origres[origres.endpoint==endpoint].iloc[0]; r=repres[repres.endpoint==endpoint].iloc[0]
    comparison += [{"endpoint":endpoint,"sample":"original_validated","estimate":o.estimate,"ci_low":o.ci_low,"ci_high":o.ci_high,"n_patch_years":o.n_patch_years,"n_patches":o.n_patches,"n_blocks":o.n_blocks},
                   {"endpoint":endpoint,"sample":"independent_replication","estimate":r.estimate,"ci_low":r.ci_low,"ci_high":r.ci_high,"n_patch_years":r.n_patch_years,"n_patches":r.n_patches,"n_blocks":r.n_blocks}]
pd.DataFrame(comparison).to_csv(TABLES/"original_vs_replication.csv",index=False,encoding="utf-8-sig")

o=pd.read_csv(TABLES/"validated_primary_sample.csv"); o=o[o.sensor_strategy=="C_platform_aware"].copy(); o["sample_origin"]="original_validated"
r=pd.read_csv(OUT/"replication_primary_sample.csv");
cols=["sample_origin","patch_id","block_id","year","vegetation_type","built_fraction_1000m","crop_context_1000m","include_confirmatory","summer_NDVI","peak_NDVI"]
d=pd.concat([o[cols],r[cols]],ignore_index=True); d["include_confirmatory"]=d.include_confirmatory.astype(str).str.lower().eq("true")

def run(endpoint, crop=True, interaction=False):
    q=d[d.include_confirmatory].dropna(subset=[endpoint]).copy()
    rhs="built_fraction_1000m + C(vegetation_type) + C(year)" + (" + crop_context_1000m" if crop else "")
    if interaction: rhs += " + built_fraction_1000m:C(vegetation_type)"
    formula=f"{endpoint} ~ {rhs}"; m=smf.ols(formula,q).fit(cov_type="cluster",cov_kwds={"groups":q.block_id})
    term="built_fraction_1000m"; est=m.params[term]; se=m.bse[term]; blocks=q.block_id.unique(); boots=[]; loo=[]
    for _ in range(CFG["replication"]["block_bootstrap_replicates"]):
        parts=[]
        for j,b in enumerate(RNG.choice(blocks,len(blocks),replace=True)):
            z=q[q.block_id==b].copy(); z["boot_block"]=f"c{j}"; parts.append(z)
        try: boots.append(smf.ols(formula,pd.concat(parts,ignore_index=True)).fit().params[term])
        except Exception: pass
    for b in blocks:
        z=q[q.block_id!=b]
        try: loo.append({"endpoint":endpoint,"model":"crop_adjusted" if crop else "no_crop_context","left_out_block":b,"estimate":smf.ols(formula,z).fit().params[term]})
        except Exception: pass
    rec={"endpoint":endpoint,"model":"vegetation_interaction" if interaction else ("crop_adjusted" if crop else "no_crop_context"),"estimate":est,"standard_error":se,
      "ci_low":est-1.96*se,"ci_high":est+1.96*se,"p_value":m.pvalues[term],"bootstrap_ci_low":np.percentile(boots,2.5),"bootstrap_ci_high":np.percentile(boots,97.5),
      "n_patch_years":len(q),"n_patches":q.patch_id.nunique(),"n_blocks":len(blocks),"original_patch_years":sum(q.sample_origin=="original_validated"),"replication_patch_years":sum(q.sample_origin=="independent_replication"),
      "loo_negative":sum(x["estimate"]<0 for x in loo),"loo_total":len(loo),"formula":formula}
    rec["contrast_30pct_vs_5pct_built"]=est*(.30-.05)
    rec["contrast_75pct_vs_5pct_built"]=est*(.75-.05)
    if interaction:
        it=[x for x in m.params.index if "built_fraction_1000m:C(vegetation_type)" in x]
        rec["grass_vs_tree_slope_difference"]=m.params[it[0]] if it else np.nan
        rec["grass_vs_tree_interaction_p"]=m.pvalues[it[0]] if it else np.nan
    return rec,loo

models=[]; loos=[]
for endpoint in ["summer_NDVI","peak_NDVI"]:
    for crop,inter in [(True,False),(False,False),(True,True)]:
        rec,lo=run(endpoint,crop,inter); models.append(rec)
        if crop and not inter: loos.extend(lo)
pd.DataFrame(models).to_csv(TABLES/"combined_primary_models.csv",index=False,encoding="utf-8-sig")
pd.DataFrame(loos).to_csv(TABLES/"combined_leave_one_block_out.csv",index=False,encoding="utf-8-sig")
print(pd.DataFrame(models).to_string(index=False))

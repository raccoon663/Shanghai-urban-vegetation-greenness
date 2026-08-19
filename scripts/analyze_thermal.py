"""Secondary Landsat LST validation on fixed original and replication blocks."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,time,requests
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; DATA=ROOT/"data"/"processed"
TABLES = ROOT / "outputs" / "tables"
CFG=json.loads((ROOT/"config/final_analysis.json").read_text(encoding="utf-8")); P=CFG["analysis"]
oval=pd.read_csv(TABLES/"patch_validation.csv"); oval=oval[oval.include_confirmatory.astype(str).str.lower().eq("true")]
op=pd.read_csv(OUT/"sampled_patches.csv"); ob=pd.read_csv(OUT/"sampled_blocks.csv")
valid_oblocks=set(oval.block_id); ob=ob[ob.block_id.isin(valid_oblocks)].copy(); ob["sample_origin"]="original_validated"
rval=pd.read_csv(OUT/"replication_patch_validation.csv"); rval=rval[rval.include_replication_confirmatory.astype(str).str.lower().eq("true")]
rb=pd.read_csv(TABLES/"replication_blocks.csv"); rb=rb[rb.block_id.isin(set(rval.block_id))].copy(); rb["sample_origin"]="independent_replication"
blocks=pd.concat([ob,rb],ignore_index=True).drop_duplicates(["sample_origin","block_id"])
bbox=[blocks.centroid_lon.min()-.02,blocks.centroid_lat.min()-.02,blocks.centroid_lon.max()+.02,blocks.centroid_lat.max()+.02]

def search(year):
    body={"collections":["landsat-c2-l2"],"bbox":bbox,"datetime":f"{year}-06-01/{year}-08-31","limit":1000,"query":{"eo:cloud_cover":{"lt":70}}}
    for a in range(4):
        try:
            r=requests.post(f'{CFG["stac_url"]}/search',json=body,timeout=40); r.raise_for_status(); return r.json().get("features",[])
        except Exception:
            if a==3: raise
            time.sleep(2+a*2)
def get(item,b):
    url=f"https://planetarycomputer.microsoft.com/api/data/v1/item/point/{b.centroid_lon},{b.centroid_lat}"
    q=[("collection","landsat-c2-l2"),("item",item["id"]),("assets","qa_pixel"),("assets","lwir11")]
    for a in range(3):
        try:
            r=requests.get(url,params=q,timeout=30,headers={"User-Agent":"ShanghaiUrbanVegetation/1.0"}); r.raise_for_status(); qa,dn=r.json()["values"]; qa=int(qa); dn=float(dn)
            bad=any(qa&(1<<bit) for bit in CFG["landsat_qa_exclude_bits"]); temp=dn*CFG["thermal_analysis"]["landsat_st_scale"]+CFG["thermal_analysis"]["landsat_st_offset_kelvin"]-273.15
            dt=pd.Timestamp(item["properties"]["datetime"])
            return {"sample_origin":b.sample_origin,"block_id":b.block_id,"year":dt.year,"date":dt.date(),"item_id":item["id"],"platform":item["properties"].get("platform"),"built_fraction":b.built_fraction,"qa_pixel":qa,"lst_c":temp if not bad and -10<=temp<=70 else np.nan}
        except Exception:
            if a==2:return None
            time.sleep(2+a*2)

allrows=[]
for year in P["years"]:
    f=DATA/f"replication_thermal_points_{year}.csv"
    if f.exists(): allrows.extend(pd.read_csv(f).to_dict("records")); continue
    items=search(year); jobs=[]
    for it in items:
        x0,y0,x1,y1=it["bbox"]
        for _,b in blocks.iterrows():
            if x0<=b.centroid_lon<=x1 and y0<=b.centroid_lat<=y1:jobs.append((it,b))
    rows=[]
    with ThreadPoolExecutor(max_workers=30) as ex:
        for fu in as_completed([ex.submit(get,it,b) for it,b in jobs]):
            z=fu.result()
            if z:rows.append(z)
    pd.DataFrame(rows).to_csv(f,index=False); allrows.extend(rows); print(year,len(rows),flush=True)
raw=pd.DataFrame(allrows); raw.to_csv(DATA/"replication_thermal_block_scene_points.csv",index=False)
agg=raw.groupby(["sample_origin","block_id","year"],as_index=False).agg(summer_median_lst_c=("lst_c","median"),n_valid_summer_scenes=("lst_c","count"),built_fraction=("built_fraction","first"))
q=agg[agg.n_valid_summer_scenes>=2].dropna().copy(); m=smf.ols("summer_median_lst_c ~ built_fraction + C(year) + C(sample_origin)",q).fit(cov_type="cluster",cov_kwds={"groups":q.block_id})
est=m.params["built_fraction"]; se=m.bse["built_fraction"]
summary=pd.DataFrame([{"analysis":"combined validated blocks; secondary thermal","estimate_c_per_built_fraction":est,"standard_error":se,"ci_low":est-1.96*se,"ci_high":est+1.96*se,"p_value":m.pvalues["built_fraction"],"n_block_years":len(q),"n_blocks":q.block_id.nunique(),"original_blocks":q[q.sample_origin=="original_validated"].block_id.nunique(),"replication_blocks":q[q.sample_origin=="independent_replication"].block_id.nunique()}])
agg.merge(summary.assign(_k=1),how="cross").drop(columns="_k",errors="ignore").to_csv(TABLES/"thermal_validation.csv",index=False,encoding="utf-8-sig")
print(summary.to_string(index=False))

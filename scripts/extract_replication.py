"""Extract real S2/Landsat observations for fixed validated replication patches."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, time, requests
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; DATA=ROOT/"data"/"processed"
TABLES = ROOT / "outputs" / "tables"
CFG=json.loads((ROOT/"config/final_analysis.json").read_text(encoding="utf-8")); P=CFG["analysis"]
validation=pd.read_csv(OUT/"replication_patch_validation.csv")
keep=validation[validation.include_replication_confirmatory.astype(str).str.lower().eq("true")]
pts=pd.read_csv(DATA/"replication_patch_sample_points.csv"); pts=pts[pts.patch_id.isin(keep.patch_id)]
bbox=[pts.lon.min()-.02,pts.lat.min()-.02,pts.lon.max()+.02,pts.lat.max()+.02]

def in_ring(x,y,ring):
    inside=False; j=len(ring)-1
    for i in range(len(ring)):
        xi,yi=ring[i][:2]; xj,yj=ring[j][:2]
        if ((yi>y)!=(yj>y)) and x < (xj-xi)*(y-yi)/(yj-yi)+xi: inside=not inside
        j=i
    return inside
def covered(geometry,x,y):
    polys=geometry["coordinates"] if geometry["type"]=="MultiPolygon" else [geometry["coordinates"]]
    return any(in_ring(x,y,p[0]) and not any(in_ring(x,y,h) for h in p[1:]) for p in polys)

def search(collection,year,cloud):
    body={"collections":[collection],"bbox":bbox,"datetime":f"{year}-01-01/{year}-12-31","limit":1000,"query":{"eo:cloud_cover":{"lt":cloud}}}
    for a in range(5):
        try:
            r=requests.post(f'{CFG["stac_url"]}/search',json=body,timeout=40); r.raise_for_status(); return r.json().get("features",[])
        except Exception:
            if a==4: raise
            time.sleep(2+a*2)
def point(item,p,sensor):
    url=f"https://planetarycomputer.microsoft.com/api/data/v1/item/point/{p.lon},{p.lat}"
    assets=["B04","B08","SCL"] if sensor=="S2" else ["qa_pixel","red","nir08"]
    params=[("collection","sentinel-2-l2a" if sensor=="S2" else "landsat-c2-l2"),("item",item["id"]),*[("assets",x) for x in assets]]
    for a in range(4):
        try:
            r=requests.get(url,params=params,timeout=30,headers={"User-Agent":"ShanghaiUrbanVegetation/1.0"}); r.raise_for_status(); v=r.json()["values"]
            dt=pd.Timestamp(item["properties"]["datetime"])
            dt_cmp=dt.tz_localize(None) if dt.tzinfo is not None else dt
            if sensor=="S2":
                red,nir,scl=v; red*=1e-4; nir*=1e-4
                if dt_cmp>=pd.Timestamp(P["sentinel2_baseline_offset_start"]): red-=.1; nir-=.1
                clear=int(scl) in CFG["sentinel2_clear_scl"] and red>0 and nir>0
                return {**p.to_dict(),"date":dt.date(),"year":dt.year,"item_id":item["id"],"platform":item["properties"].get("platform"),"red":red,"nir":nir,"qa":int(scl),"clear":clear},None
            qa,rd,nd=v; qa=int(qa); red=float(rd)*.0000275-.2; nir=float(nd)*.0000275-.2
            clear=not any(qa&(1<<b) for b in CFG["landsat_qa_exclude_bits"]) and red>0 and nir>0
            return {**p.to_dict(),"date":dt.date(),"year":dt.year,"item_id":item["id"],"platform":item["properties"].get("platform"),"red":red,"nir":nir,"qa":qa,"clear":clear},None
        except Exception as e:
            if a==3:return None,str(e)
            time.sleep(2+a*2)

def extract(sensor):
    all_patch=[]; inv=[]
    for year in P["years"]:
        final=DATA/f"replication_{sensor.lower()}_points_{year}.csv"
        patchfile=DATA/f"replication_{sensor.lower()}_patch_{year}.csv"
        if final.exists() and patchfile.exists():
            d=pd.read_csv(patchfile); all_patch.extend(d.to_dict("records")); inv.append({"sensor":sensor,"year":year,"reused":True,"valid_patch_dates":len(d)}); continue
        raw=search("sentinel-2-l2a" if sensor=="S2" else "landsat-c2-l2",year,60 if sensor=="S2" else 70); best={}
        for it in raw:
            dt=pd.Timestamp(it["properties"]["datetime"]); pr=it["properties"]
            key=(pr.get("s2:mgrs_tile","unknown"),(dt.dayofyear-1)//10) if sensor=="S2" else (pr.get("landsat:wrs_path"),pr.get("landsat:wrs_row"),(dt.dayofyear-1)//10)
            cloud=pr.get("eo:cloud_cover",100)
            if key not in best or cloud<best[key][0]:best[key]=(cloud,it)
        jobs=[]
        for it in [x[1] for x in best.values()]:
            # Shanghai crosses adjacent MGRS/path-row footprints. A bbox test alone
            # assigns points to scenes whose raster footprint does not cover them.
            geom=it["geometry"]
            inside=pts[pts.apply(lambda p: covered(geom,p.lon,p.lat),axis=1)]
            jobs.extend((it,p) for _,p in inside.iterrows())
        checkpoint=DATA/f"replication_{sensor.lower()}_points_{year}.partial.csv"
        if checkpoint.exists(): rows=pd.read_csv(checkpoint).to_dict("records")
        else: rows=[]
        completed={(str(r["item_id"]),str(r["sample_point_id"])) for r in rows}
        jobs=[(it,p) for it,p in jobs if (str(it["id"]),str(p.sample_point_id)) not in completed]
        errors=[]
        with ThreadPoolExecutor(max_workers=36) as ex:
            fs=[ex.submit(point,it,p,sensor) for it,p in jobs]
            for n,f in enumerate(as_completed(fs),1):
                row,err=f.result();
                if row:rows.append(row)
                if err:errors.append(err)
                if n%100==0 or n==len(fs):
                    pd.DataFrame(rows).to_csv(checkpoint,index=False)
                    print(sensor,year,n,len(fs),"rows",len(rows),"errors",len(errors),flush=True)
        d=pd.DataFrame(rows)
        if d.empty:
            raise RuntimeError(f"{sensor} {year}: zero successful point rows from {len(jobs)} exact-footprint jobs; first errors={errors[:3]}")
        d=d.drop_duplicates(["item_id","sample_point_id"]); d.to_csv(final,index=False)
        prows=[]
        for (pid,date,item),g in d.groupby(["patch_id","date","item_id"]):
            good=g[g.clear.astype(str).str.lower().eq("true")]
            if len(good)>=P["minimum_clear_sample_points"]:
                base=g.iloc[0]; red=good.red.median(); nir=good.nir.median()
                prows.append({"patch_id":pid,"block_id":base.block_id,"vegetation_type":base.vegetation_type,"date":date,"year":year,"item_id":item,"platform":base.platform,"valid_sample_points":len(good),"ndvi_raw":(nir-red)/(nir+red)})
        pd.DataFrame(prows).to_csv(patchfile,index=False); all_patch.extend(prows); inv.append({"sensor":sensor,"year":year,"raw":len(raw),"selected":len(best),"point_rows":len(d),"errors":len(errors),"valid_patch_dates":len(prows)})
        print(inv[-1],flush=True)
    return pd.DataFrame(all_patch),inv

s2,si=extract("S2"); ls,li=extract("Landsat")
s2=s2.rename(columns={"ndvi_raw":"ndvi"}); s2["date"]=pd.to_datetime(s2.date); ls["date"]=pd.to_datetime(ls.date)
s2.to_csv(DATA/"replication_s2_timeseries.csv",index=False); ls.to_csv(DATA/"replication_landsat_raw.csv",index=False)
pd.DataFrame(si+li).to_csv(TABLES/"replication_sensor_inventory.csv",index=False,encoding="utf-8-sig")

# Platform-aware L8/L9 gap fills, using fixed original-sample platform coefficients.
cal=pd.read_csv(TABLES/"platform_calibration.csv"); coefs={(r.s2_platform,r.landsat_platform):(r.intercept,r.slope,r.tree_offset) for r in cal.itertuples(index=False) if r.landsat_platform in ("Landsat 8","Landsat 9") and .5<=r.slope<=1.5 and r.rmse<=.18}
ls["landsat_platform"]=ls.item_id.str.extract(r"^(L[CET]0[789])").replace({"LC08":"Landsat 8","LC09":"Landsat 9","LE07":"Landsat 7"})
s2["s2_platform"]=s2.item_id.str.extract(r"^(S2[AB])")
fill=[]
for lr in ls.itertuples(index=False):
    if lr.landsat_platform not in ("Landsat 8","Landsat 9"):continue
    g=s2[(s2.patch_id==lr.patch_id)&(s2.year==lr.year)].copy()
    if g.empty:continue
    g["dd"]=(g.date-lr.date).abs().dt.days; n=g.sort_values("dd").iloc[0]
    if n.dd<=3:continue
    key=(n.s2_platform,lr.landsat_platform)
    if key not in coefs:continue
    a,b,t=coefs[key]; fill.append({"patch_id":lr.patch_id,"block_id":lr.block_id,"vegetation_type":lr.vegetation_type,"date":lr.date,"year":lr.year,"item_id":lr.item_id,"platform":lr.platform,"valid_sample_points":lr.valid_sample_points,"ndvi":np.clip(a+b*lr.ndvi_raw+t*(lr.vegetation_type=="tree"),-.2,1),"source":"Landsat platform-aware validated calibration"})
s2["source"]="Sentinel-2 PB04-harmonized"
combined=pd.concat([s2[["patch_id","block_id","vegetation_type","date","year","item_id","platform","valid_sample_points","ndvi","source"]],pd.DataFrame(fill)],ignore_index=True).sort_values(["patch_id","date"])
combined.to_csv(DATA/"replication_combined_timeseries.csv",index=False)
print({"included_patches":keep.shape[0],"s2_dates":len(s2),"landsat_dates":len(ls),"platform_aware_fill_dates":len(fill)})

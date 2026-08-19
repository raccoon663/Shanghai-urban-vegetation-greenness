"""Freeze new independent replication blocks and patches without reading NDVI outcomes."""
from pathlib import Path
import json, math

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.windows import from_bounds
from scipy.ndimage import label

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"
TABLES = ROOT / "outputs" / "tables"
CFG=json.loads((ROOT/"config/final_analysis.json").read_text(encoding="utf-8")); P=CFG["replication"]
candidates=json.loads((ROOT/"data/interim/candidate_blocks.geojson").read_text(encoding="utf-8"))["features"]
old=pd.read_csv(OUT/"sampled_blocks.csv")

def bbox(f):
    q=f["geometry"]["coordinates"][0]; return min(x[0] for x in q),min(x[1] for x in q),max(x[0] for x in q),max(x[1] for x in q)
def dist(a,b):
    return math.hypot((a[0]-b[0])*111320*math.cos(math.radians((a[1]+b[1])/2)),(a[1]-b[1])*111320)
def comps(a,tr,code):
    lab,n=label(a==code,structure=np.ones((3,3),int)); z=[]; sizes=np.bincount(lab.ravel())
    for lid in range(1,len(sizes)):
        if sizes[lid] < 9: continue
        yy,xx=np.where(lab==lid); xy=np.array([rasterio.transform.xy(tr,int(y),int(x)) for y,x in zip(yy,xx)])
        cen=xy.mean(axis=0); z.append({"count":int(sizes[lid]),"yy":yy,"xx":xx,"xy":xy,"centroid":(float(cen[0]),float(cen[1])),"mask":lab==lid})
    return sorted(z,key=lambda v:(-v["count"],round(v["centroid"][0],8),round(v["centroid"][1],8)))

metrics=[]; components={}; codes={"tree":10,"grass":30,"crop":40,"built":50,"water":80}
with rasterio.open(ROOT/"data/interim/worldcover_shanghai_30m.tif") as ds:
    for f in candidates:
        bid=f["properties"]["block_id"]
        if bid in set(old.block_id): continue
        w=from_bounds(*bbox(f),transform=ds.transform).round_offsets().round_lengths()
        a=ds.read(1,window=w,boundless=True,fill_value=0); tr=ds.window_transform(w); n=a.size
        r={"block_id":bid,"centroid_lon":f["properties"]["center_lon"],"centroid_lat":f["properties"]["center_lat"],"area_km2":25.0}
        for name,code in codes.items(): r[name+"_fraction"]=float((a==code).sum()/n)
        r["vegetation_fraction"]=r["tree_fraction"]+r["grass_fraction"]
        tc,gc=comps(a,tr,10),comps(a,tr,30); components[bid]={"tree":tc,"grass":gc,"transform":tr}
        r["eligible_tree_components"]=len(tc); r["eligible_grass_components"]=len(gc)
        metrics.append(r)
md=pd.DataFrame(metrics)
edges=[-1,.20,.50,2]; names=["low","medium","high"]
md["urbanization_stratum"]=pd.cut(md.built_fraction,edges,labels=names).astype(str)
md=md[(md.water_fraction<.4)&(md.vegetation_fraction>=.05)&((md.eligible_tree_components>0)|(md.eligible_grass_components>0))].copy()

# Exclude any candidate within 6 km of an original block. Select 15, balanced 5/stratum,
# closest to fixed stratum targets and spatially separated. No NDVI file is read.
oldxy=list(zip(old.centroid_lon,old.centroid_lat)); chosen=[]
targets={"low":.10,"medium":.35,"high":.70}
for st in names:
    d=md[md.urbanization_stratum==st].copy(); d["rank"]=(d.built_fraction-targets[st]).abs()
    for row in d.sort_values(["rank","block_id"]).itertuples(index=False):
        xy=(row.centroid_lon,row.centroid_lat)
        if min(dist(xy,z) for z in oldxy) < P["minimum_block_separation_m"]: continue
        if any(dist(xy,(q["centroid_lon"],q["centroid_lat"])) < P["minimum_block_separation_m"] for q in chosen): continue
        chosen.append(row._asdict())
        if sum(q["urbanization_stratum"]==st for q in chosen)>=5: break
blocks=pd.DataFrame(chosen)
if len(blocks)<P["replication_target_blocks_min"]: raise RuntimeError(f"Only {len(blocks)} replication blocks")

sel=set(blocks.block_id); blockgeo=[]; patchrows=[]; patchgeo=[]; samples=[]
with rasterio.open(ROOT/"data/interim/worldcover_shanghai_30m.tif") as ds:
    for f in candidates:
        bid=f["properties"]["block_id"]
        if bid not in sel: continue
        br=blocks[blocks.block_id==bid].iloc[0]; props={k:(v.item() if hasattr(v,"item") else v) for k,v in br.items() if k!="rank"}
        f["properties"].update(props); blockgeo.append(f); picks=[]
        for vt,code in [("tree",10),("grass",30)]:
            for z in components[bid][vt]:
                if all(dist(z["centroid"],q["centroid"])>=300 for q in picks): picks.append({**z,"vegetation_type":vt,"code":code}); break
        for z in picks:
            vt=z["vegetation_type"]; lon,lat=z["centroid"]; yy,xx=z["yy"],z["xx"]
            pixw=abs(ds.transform.a)*111320*math.cos(math.radians(lat)); pixh=abs(ds.transform.e)*111320
            vals={}
            for rad in [90,1000]:
                dlat=rad/111320; dlon=rad/(111320*math.cos(math.radians(lat)))
                w=from_bounds(lon-dlon,lat-dlat,lon+dlon,lat+dlat,ds.transform).round_offsets().round_lengths()
                a=ds.read(1,window=w,boundless=True,fill_value=0)
                vals[rad]={"built":float((a==50).mean()),"crop":float((a==40).mean()),"same":float((a==z["code"]).mean())}
            pid=f"{bid}_{vt}_R1"; purity=vals[90]["same"]
            conf="HIGH" if purity>=2/3 else ("MEDIUM" if purity>=1/3 else "LOW")
            pr={"patch_id":pid,"block_id":bid,"vegetation_type":vt,"patch_area_ha":z["count"]*pixw*pixh/10000,
                "centroid_lon":lon,"centroid_lat":lat,"built_fraction_1000m":vals[1000]["built"],
                "built_fraction_90m":vals[90]["built"],"crop_context_1000m":vals[1000]["crop"],
                "worldcover_local_purity_90m":purity,"validation_confidence":conf,
                "selection_basis":"WorldCover class/component area/purity; no NDVI outcomes"}
            gg=next(g for g,val in shapes(z["mask"].astype("uint8"),mask=z["mask"],transform=components[bid]["transform"]) if val==1)
            patchrows.append(pr); patchgeo.append({"type":"Feature","properties":pr,"geometry":gg})
            xy=z["xy"]; u=xy-xy.mean(axis=0); _,_,v=np.linalg.svd(u,full_matrices=False); score=u@v[0]; order=np.argsort(score)
            for k,q in enumerate([.25,.5,.75],1):
                x,y=xy[order[int((len(order)-1)*q)]]; samples.append({"patch_id":pid,"block_id":bid,"vegetation_type":vt,"sample_point_id":f"{pid}_p{k}","lon":x,"lat":y})

blocks=blocks.drop(columns=["rank"],errors="ignore")
blocks.to_csv(TABLES/"replication_blocks.csv",index=False,encoding="utf-8-sig")
pd.DataFrame(patchrows).to_csv(TABLES/"replication_patches.csv",index=False,encoding="utf-8-sig")
pd.DataFrame(samples).to_csv(ROOT/"data/processed/replication_patch_sample_points.csv",index=False,encoding="utf-8-sig")
(OUT/"replication_blocks.geojson").write_text(json.dumps({"type":"FeatureCollection","features":blockgeo}),encoding="utf-8")
(OUT/"replication_patches.geojson").write_text(json.dumps({"type":"FeatureCollection","features":patchgeo}),encoding="utf-8")
print({"blocks":len(blocks),"strata":blocks.urbanization_stratum.value_counts().to_dict(),"patches":len(patchrows),"confidence":pd.DataFrame(patchrows).validation_confidence.value_counts().to_dict()})

""" Landsat summer LST, 2017-2025, and outcome-independent hot years."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, time, requests
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config/final_analysis.json").read_text(encoding="utf-8"))
P = CFG["analysis"]; OUT = ROOT / "outputs"
BLOCKS = pd.read_csv(OUT / "sampled_blocks.csv")
BBOX = [BLOCKS.centroid_lon.min() - .03, BLOCKS.centroid_lat.min() - .03,
        BLOCKS.centroid_lon.max() + .03, BLOCKS.centroid_lat.max() + .03]
LOG = []
def log(x):
    s = f"{pd.Timestamp.now().isoformat()} {x}"; print(s, flush=True); LOG.append(s)
def search(year):
    body = {"collections": ["landsat-c2-l2"], "bbox": BBOX,
            "datetime": f"{year}-06-01/{year}-08-31", "limit": 1000,
            "query": {"eo:cloud_cover": {"lt": 70}}}
    for a in range(4):
        try:
            r = requests.post(f'{CFG["stac_url"]}/search', json=body, timeout=30); r.raise_for_status()
            return r.json().get("features", [])
        except Exception:
            if a == 3: raise
            time.sleep(2 + a * 2)
def task(item, block):
    url = f"https://planetarycomputer.microsoft.com/api/data/v1/item/point/{block.centroid_lon},{block.centroid_lat}"
    params = [("collection", "landsat-c2-l2"), ("item", item["id"]),
              ("assets", "qa_pixel"), ("assets", "lwir11")]
    for a in range(3):
        try:
            r = requests.get(url, params=params, timeout=25,
                             headers={"User-Agent": "ShanghaiUrbanVegetation/1.0"}); r.raise_for_status()
            qa, dn = r.json()["values"]; qa, dn = int(qa), float(dn)
            bad = any(qa & (1 << bit) for bit in CFG["landsat_qa_exclude_bits"])
            temp = dn * CFG["thermal_analysis"]["landsat_st_scale"] + CFG["thermal_analysis"]["landsat_st_offset_kelvin"] - 273.15
            valid = (not bad) and np.isfinite(temp) and -10 <= temp <= 70
            dt = pd.Timestamp(item["properties"]["datetime"])
            return {"block_id": block.block_id, "year": dt.year, "date": dt.date(),
                    "item_id": item["id"], "platform": item["properties"].get("platform"),
                    "scene_cloud_percent": item["properties"].get("eo:cloud_cover"),
                    "qa_pixel": qa, "lst_dn": dn, "lst_c": temp if valid else np.nan,
                    "valid": valid}, None
        except Exception as exc:
            if a == 2: return None, (item["id"], block.block_id, str(exc))
            time.sleep(2 + a * 2)

all_rows = []; inventory = []
for year in P["years"]:
    items = search(year)
    final = ROOT / f"data/processed/thermal_points_{year}.csv"
    invfile = OUT / f"thermal_inventory_{year}.csv"
    if final.exists() and invfile.exists():
        ydf = pd.read_csv(final); all_rows.extend(ydf.to_dict("records")); inventory.append(pd.read_csv(invfile).iloc[0].to_dict())
        log(f"{year}: reused completed thermal year ({len(ydf)} rows)"); continue
    checkpoint = ROOT / f"data/processed/thermal_points_{year}.partial.csv"
    if checkpoint.exists(): rows = pd.read_csv(checkpoint).to_dict("records")
    else:
        old = ROOT / f"data/processed/sampled_thermal_points_{year}.csv"
        rows = pd.read_csv(old).to_dict("records") if old.exists() else []
        if rows: pd.DataFrame(rows).to_csv(checkpoint, index=False)
    completed = {(str(r["item_id"]), str(r["block_id"])) for r in rows}; jobs = []
    for item in items:
        ib = item["bbox"]
        for _, block in BLOCKS.iterrows():
            if (str(item["id"]), str(block.block_id)) not in completed and ib[0] <= block.centroid_lon <= ib[2] and ib[1] <= block.centroid_lat <= ib[3]:
                jobs.append((item, block))
    log(f"{year}: scenes={len(items)}, resumed={len(rows)}, remaining={len(jobs)}"); errors = []
    with ThreadPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(task, item, block) for item, block in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            row, error = future.result()
            if row is not None: rows.append(row)
            if error: errors.append(error)
            if n % 100 == 0 or n == len(futures): pd.DataFrame(rows).to_csv(checkpoint, index=False)
    ydf = pd.DataFrame(rows).drop_duplicates(["item_id", "block_id"]); ydf.to_csv(final, index=False)
    inv = {"year": year, "candidate_scenes": len(items), "block_scene_rows": len(ydf),
           "valid_block_scene_rows": int(ydf.valid.astype(str).str.lower().eq("true").sum()),
           "failed_new_requests": len(errors)}
    pd.DataFrame([inv]).to_csv(invfile, index=False); all_rows.extend(ydf.to_dict("records")); inventory.append(inv)
points = pd.DataFrame(all_rows); points.to_csv(ROOT / "data/processed/thermal_block_scene_points.csv", index=False)
valid = points[points.valid.astype(str).str.lower().eq("true")]; agg = []
for _, block in BLOCKS.iterrows():
    for year in P["years"]:
        g = valid[(valid.block_id == block.block_id) & (valid.year == year)]
        agg.append({"block_id": block.block_id, "year": year,
                    "urbanization_stratum": block.urbanization_stratum,
                    "block_built_fraction": block.built_fraction,
                    "summer_median_lst_c": g.lst_c.median(), "summer_mean_lst_c": g.lst_c.mean(),
                    "summer_upper_quartile_lst_c": g.lst_c.quantile(.75),
                    "n_valid_summer_scenes": len(g)})
thermal = pd.DataFrame(agg); thermal.to_csv(OUT / "thermal_metrics.csv", index=False, encoding="utf-8-sig")
year_lst = thermal.groupby("year").agg(shanghai_median_summer_lst_c=("summer_median_lst_c", "median"),
    blocks_with_lst=("summer_median_lst_c", "count")).reset_index().sort_values("shanghai_median_summer_lst_c")
year_lst["thermal_year_class"] = "middle"
year_lst.loc[year_lst.index[:3], "thermal_year_class"] = "cool"
year_lst.loc[year_lst.index[-3:], "thermal_year_class"] = "hot"
year_lst["hot_year"] = year_lst.thermal_year_class.eq("hot")
year_lst.to_csv(OUT / "hot_year_classification.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(inventory).to_csv(OUT / "thermal_inventory.csv", index=False, encoding="utf-8-sig")
(ROOT / "logs/thermal_extract.log").write_text("\n".join(LOG), encoding="utf-8")
print(year_lst.sort_values("year").to_string(index=False))

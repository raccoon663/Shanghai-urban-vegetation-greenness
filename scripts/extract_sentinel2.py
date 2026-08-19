""" Sentinel-2 reference series, 2017-2025, fixed 10-day rule."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, time, requests
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config/final_analysis.json").read_text(encoding="utf-8"))
P = CFG["analysis"]; PTS = pd.read_csv(ROOT / "data/processed/sampled_patch_sample_points.csv")
PATCHES = pd.read_csv(ROOT / "outputs/sampled_patches.csv")
BBOX = [PTS.lon.min() - .02, PTS.lat.min() - .02, PTS.lon.max() + .02, PTS.lat.max() + .02]
LOG = []
def log(x, level="INFO"):
    s = f"{pd.Timestamp.now().isoformat()} [{level}] {x}"; print(s, flush=True); LOG.append(s)
def search(year):
    body = {"collections": ["sentinel-2-l2a"], "bbox": BBOX,
            "datetime": f"{year}-01-01/{year}-12-31", "limit": 1000,
            "query": {"eo:cloud_cover": {"lt": P["sentinel_scene_cloud_max_percent"]}}}
    for a in range(4):
        try:
            r = requests.post(f'{CFG["stac_url"]}/search', json=body, timeout=30); r.raise_for_status()
            return r.json().get("features", [])
        except Exception as exc:
            if a == 3: raise
            log(f"STAC retry {a + 1}: {exc}", "WARN"); time.sleep(2 + a * 2)
def task(item, point):
    url = f"https://planetarycomputer.microsoft.com/api/data/v1/item/point/{point.lon},{point.lat}"
    params = [("collection", "sentinel-2-l2a"), ("item", item["id"]),
              ("assets", "B04"), ("assets", "B08"), ("assets", "SCL")]
    for a in range(3):
        try:
            r = requests.get(url, params=params, timeout=25,
                             headers={"User-Agent": "ShanghaiUrbanVegetation/1.0"}); r.raise_for_status()
            red, nir, scl = r.json()["values"]; dt = pd.Timestamp(item["properties"]["datetime"])
            return {**point.to_dict(), "date": dt.date(), "year": dt.year, "item_id": item["id"],
                    "red": red * 1e-4, "nir": nir * 1e-4, "scl": int(scl),
                    "clear": int(scl) in CFG["sentinel2_clear_scl"], "source": "Sentinel-2 L2A"}, None
        except Exception as exc:
            if a == 2: return None, (item["id"], point.sample_point_id, f"{type(exc).__name__}: {exc}")
            time.sleep(2 + a * 2)
def seed(year):
    old = ROOT / f"data/processed/sampled_s2_point_samples_{year}.csv"
    return pd.read_csv(old) if old.exists() else pd.DataFrame()

all_patch = []; density = []; inventory = []
for year in P["years"]:
    raw = search(year); best = {}
    for item in raw:
        dt = pd.Timestamp(item["properties"]["datetime"])
        key = (item["properties"].get("s2:mgrs_tile", "unknown"), (dt.dayofyear - 1) // 10)
        cloud = item["properties"].get("eo:cloud_cover", 100)
        if key not in best or cloud < best[key][0]: best[key] = (cloud, item)
    items = [v[1] for v in best.values()]; candidate = defaultdict(set); jobs = []
    for item in items:
        ib = item["bbox"]; dt = pd.Timestamp(item["properties"]["datetime"])
        inside = PTS[(PTS.lon >= ib[0]) & (PTS.lon <= ib[2]) & (PTS.lat >= ib[1]) & (PTS.lat <= ib[3])]
        for patch_id in inside.patch_id.unique(): candidate[patch_id].add(dt.date())
        for _, point in inside.iterrows(): jobs.append((item, point))
    checkpoint = ROOT / f"data/processed/s2_point_samples_{year}.partial.csv"
    if checkpoint.exists(): rows = pd.read_csv(checkpoint).to_dict("records")
    else:
        seeded = seed(year); rows = seeded.to_dict("records")
        if rows: seeded.to_csv(checkpoint, index=False)
    completed = {(str(r["item_id"]), str(r["sample_point_id"])) for r in rows}
    jobs = [(item, point) for item, point in jobs
            if (str(item["id"]), str(point.sample_point_id)) not in completed]
    log(f"{year}: raw={len(raw)}, selected={len(items)}, resumed={len(rows)}, remaining={len(jobs)}")
    errors = []
    with ThreadPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(task, item, point) for item, point in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            row, error = future.result()
            if row is not None: rows.append(row)
            if error: errors.append(error)
            if n % 100 == 0 or n == len(futures):
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
                log(f"{year}: requests {n}/{len(futures)}, rows={len(rows)}, errors={len(errors)}")
    point_df = pd.DataFrame(rows).drop_duplicates(["item_id", "sample_point_id"])
    point_df.to_csv(ROOT / f"data/processed/s2_point_samples_{year}.csv", index=False)
    patch_rows = []
    for (patch_id, date, item_id), group in point_df.groupby(["patch_id", "date", "item_id"]):
        good = group[group.clear.astype(str).str.lower().eq("true") & (group.red > 0) & (group.nir > 0)]
        if len(good) >= P["minimum_clear_sample_points"]:
            red, nir = good.red.median(), good.nir.median(); base = group.iloc[0]
            patch_rows.append({"patch_id": patch_id, "block_id": base.block_id,
                "vegetation_type": base.vegetation_type, "date": date, "year": year,
                "item_id": item_id, "valid_sample_points": len(good),
                "ndvi": (nir - red) / (nir + red), "source": "Sentinel-2 L2A"})
    pd.DataFrame(patch_rows).to_csv(ROOT / f"data/processed/s2_patch_timeseries_{year}.csv", index=False)
    all_patch.extend(patch_rows); current = pd.DataFrame(patch_rows)
    for _, patch in PATCHES.iterrows():
        g = current[current.patch_id == patch.patch_id] if len(current) else current
        dates = sorted(pd.to_datetime(g.date).dt.date.unique()) if len(g) else []
        active = [d for d in dates if d.month in P["active_months"]]
        gaps = [(b - a).days for a, b in zip(active, active[1:])]
        density.append({"series": "S2-only", "patch_id": patch.patch_id, "block_id": patch.block_id,
                        "year": year, "vegetation_type": patch.vegetation_type,
                        "n_candidate_dates": len(candidate[patch.patch_id]),
                        "n_valid_observations": len(dates),
                        "max_active_season_gap_days": max(gaps) if gaps else np.nan,
                        "summer_observations": sum(d.month in P["summer_months"] for d in dates)})
    inventory.append({"year": year, "raw_scenes": len(raw), "selected_tile_10day_scenes": len(items),
                      "successful_point_rows": len(point_df), "failed_new_point_requests": len(errors),
                      "valid_patch_dates": len(patch_rows)})
pd.DataFrame(all_patch).to_csv(ROOT / "data/processed/s2_patch_timeseries.csv", index=False)
pd.DataFrame(density).to_csv(ROOT / "outputs/s2_observation_density.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(inventory).to_csv(ROOT / "outputs/s2_inventory.csv", index=False, encoding="utf-8-sig")
(ROOT / "logs/s2_extract.log").write_text("\n".join(LOG), encoding="utf-8")
print(pd.DataFrame(inventory).to_string(index=False))

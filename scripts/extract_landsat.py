""" Landsat C2 L2 real gap fill and full-period sensor calibration."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, time, requests
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config/final_analysis.json").read_text(encoding="utf-8"))
P = CFG["analysis"]; OUT = ROOT / "outputs"
PTS = pd.read_csv(ROOT / "data/processed/sampled_patch_sample_points.csv")
PATCHES = pd.read_csv(OUT / "sampled_patches.csv")
BLOCKS = pd.read_csv(OUT / "sampled_blocks.csv")
STRATUM = BLOCKS.set_index("block_id").urbanization_stratum.to_dict()
S2 = pd.read_csv(ROOT / "data/processed/s2_patch_timeseries_harmonized.csv")
BBOX = [PTS.lon.min() - .02, PTS.lat.min() - .02, PTS.lon.max() + .02, PTS.lat.max() + .02]
LOG = []
def log(x, level="INFO"):
    s = f"{pd.Timestamp.now().isoformat()} [{level}] {x}"; print(s, flush=True); LOG.append(s)
def search(year):
    body = {"collections": ["landsat-c2-l2"], "bbox": BBOX,
            "datetime": f"{year}-03-01/{year}-11-30", "limit": 1000,
            "query": {"eo:cloud_cover": {"lt": P["landsat_scene_cloud_max_percent"]}}}
    for a in range(4):
        try:
            r = requests.post(f'{CFG["stac_url"]}/search', json=body, timeout=30); r.raise_for_status()
            return r.json().get("features", [])
        except Exception as exc:
            if a == 3: raise
            log(f"STAC retry {a + 1}: {exc}", "WARN"); time.sleep(2 + a * 2)
def item_key(item):
    pr = item["properties"]; dt = pd.Timestamp(pr["datetime"])
    path = pr.get("landsat:wrs_path", pr.get("wrs:path", "unknown"))
    row = pr.get("landsat:wrs_row", pr.get("wrs:row", "unknown"))
    return str(path), str(row), int((dt.dayofyear - 1) // 10)
def task(item, point):
    url = f"https://planetarycomputer.microsoft.com/api/data/v1/item/point/{point.lon},{point.lat}"
    params = [("collection", "landsat-c2-l2"), ("item", item["id"]),
              ("assets", "qa_pixel"), ("assets", "red"), ("assets", "nir08")]
    for a in range(3):
        try:
            r = requests.get(url, params=params, timeout=25,
                             headers={"User-Agent": "ShanghaiUrbanVegetation/1.0"}); r.raise_for_status()
            qa, red_dn, nir_dn = r.json()["values"]; qa = int(qa)
            red = float(red_dn) * 0.0000275 - 0.2; nir = float(nir_dn) * 0.0000275 - 0.2
            bad = any(qa & (1 << bit) for bit in CFG["landsat_qa_exclude_bits"])
            valid = (not bad) and np.isfinite(red) and np.isfinite(nir) and red > 0 and nir > 0
            dt = pd.Timestamp(item["properties"]["datetime"])
            return {**point.to_dict(), "date": dt.date(), "year": dt.year,
                    "item_id": item["id"], "platform": item["properties"].get("platform"),
                    "scene_cloud_percent": item["properties"].get("eo:cloud_cover"),
                    "qa_pixel": qa, "red": red, "nir": nir, "clear": valid,
                    "source": "Landsat C2 L2 SR"}, None
        except Exception as exc:
            if a == 2: return None, (item["id"], point.sample_point_id, f"{type(exc).__name__}: {exc}")
            time.sleep(2 + a * 2)

all_patch = []; inventory = []
for year in P["years"]:
    raw = search(year); best = {}
    for item in raw:
        key = item_key(item); cloud = item["properties"].get("eo:cloud_cover", 100)
        if key not in best or cloud < best[key][0]: best[key] = (cloud, item)
    items = [v[1] for v in best.values()]
    final_points = ROOT / f"data/processed/landsat_point_samples_{year}.csv"
    final_patch = ROOT / f"data/processed/landsat_patch_timeseries_{year}.csv"
    final_inv = OUT / f"landsat_inventory_{year}.csv"
    if final_points.exists() and final_patch.exists() and final_inv.exists():
        point_df = pd.read_csv(final_points); patch_df = pd.read_csv(final_patch)
        inv = pd.read_csv(final_inv).iloc[0].to_dict(); all_patch.extend(patch_df.to_dict("records")); inventory.append(inv)
        log(f"{year}: reused completed  Landsat year ({len(point_df)} point rows)"); continue
    checkpoint = ROOT / f"data/processed/landsat_point_samples_{year}.partial.csv"
    if checkpoint.exists(): rows = pd.read_csv(checkpoint).to_dict("records")
    else:
        old = ROOT / f"data/processed/sampled_landsat_point_samples_{year}.csv"
        rows = pd.read_csv(old).to_dict("records") if old.exists() else []
        if rows: pd.DataFrame(rows).to_csv(checkpoint, index=False)
    completed = {(str(r["item_id"]), str(r["sample_point_id"])) for r in rows}; jobs = []
    for item in items:
        ib = item["bbox"]
        inside = PTS[(PTS.lon >= ib[0]) & (PTS.lon <= ib[2]) & (PTS.lat >= ib[1]) & (PTS.lat <= ib[3])]
        for _, point in inside.iterrows():
            if (str(item["id"]), str(point.sample_point_id)) not in completed: jobs.append((item, point))
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
    point_df.to_csv(final_points, index=False); patch_rows = []
    for (patch_id, date, item_id), group in point_df.groupby(["patch_id", "date", "item_id"]):
        good = group[group.clear.astype(str).str.lower().eq("true")]
        if len(good) >= P["minimum_clear_sample_points"]:
            red, nir = good.red.median(), good.nir.median(); base = group.iloc[0]
            patch_rows.append({"patch_id": patch_id, "block_id": base.block_id,
                "vegetation_type": base.vegetation_type, "date": date, "year": year,
                "item_id": item_id, "valid_sample_points": len(good),
                "ndvi_raw": (nir - red) / (nir + red), "source": "Landsat C2 L2 SR"})
    patch_df = pd.DataFrame(patch_rows); patch_df.to_csv(final_patch, index=False)
    inv = {"year": year, "raw_scenes": len(raw), "selected_pathrow_10day_scenes": len(items),
           "successful_point_rows": len(point_df), "failed_new_point_requests": len(errors),
           "valid_patch_dates": len(patch_df)}
    pd.DataFrame([inv]).to_csv(final_inv, index=False); inventory.append(inv); all_patch.extend(patch_rows)

landsat = pd.DataFrame(all_patch); landsat.to_csv(ROOT / "data/processed/landsat_patch_timeseries_raw.csv", index=False)
pd.DataFrame(inventory).to_csv(OUT / "landsat_inventory.csv", index=False, encoding="utf-8-sig")
S2["date"] = pd.to_datetime(S2.date); landsat["date"] = pd.to_datetime(landsat.date)
pairs = []
for _, lr in landsat.iterrows():
    sg = S2[(S2.patch_id == lr.patch_id) & (S2.year == lr.year)].copy()
    if sg.empty: continue
    sg["delta_days"] = (sg.date - lr.date).abs().dt.days; nearest = sg.sort_values("delta_days").iloc[0]
    if nearest.delta_days <= P["landsat_s2_calibration_window_days"]:
        pairs.append({"patch_id": lr.patch_id, "year": lr.year, "vegetation_type": lr.vegetation_type,
                      "landsat_ndvi": lr.ndvi_raw, "s2_ndvi": nearest.ndvi,
                      "delta_days": int(nearest.delta_days)})
pairs = pd.DataFrame(pairs)
if len(pairs) < 10: raise RuntimeError(f"Only {len(pairs)} calibration pairs")
def fit_cal(q):
    X = np.column_stack([np.ones(len(q)), q.landsat_ndvi, (q.vegetation_type == "tree").astype(float)])
    coef = np.linalg.lstsq(X, q.s2_ndvi.to_numpy(), rcond=None)[0]; pred = X @ coef
    return coef, float(np.sqrt(np.mean((q.s2_ndvi - pred) ** 2)))
coef, rmse = fit_cal(pairs); pairs["predicted_s2_ndvi"] = coef[0] + coef[1] * pairs.landsat_ndvi + coef[2] * (pairs.vegetation_type == "tree")
pairs["residual"] = pairs.s2_ndvi - pairs.predicted_s2_ndvi
pairs.to_csv(OUT / "landsat_s2_calibration_pairs.csv", index=False, encoding="utf-8-sig")
annual = []
for year in P["years"]:
    q = pairs[pairs.year == year]
    if len(q) >= 10:
        c, r = fit_cal(q); annual.append({"year": year, "n_pairs": len(q), "intercept": c[0],
            "landsat_slope": c[1], "tree_offset": c[2], "rmse": r,
            "absolute_slope_difference_from_pooled": abs(c[1] - coef[1]), "rmse_ratio_to_pooled": r / rmse,
            "calibration_instability_flag": abs(c[1] - coef[1]) > P["calibration_instability_absolute_slope_difference"] or r / rmse > P["calibration_instability_rmse_ratio"]})
    else: annual.append({"year": year, "n_pairs": len(q), "calibration_instability_flag": True})
pd.DataFrame([{ "scope": "pooled_2017_2025", "n_pairs": len(pairs), "intercept": coef[0],
    "landsat_slope": coef[1], "tree_offset": coef[2], "rmse": rmse}]).to_csv(
    OUT / "landsat_s2_calibration.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(annual).to_csv(OUT / "landsat_s2_calibration_by_year.csv", index=False, encoding="utf-8-sig")
landsat["ndvi"] = np.clip(coef[0] + coef[1] * landsat.ndvi_raw + coef[2] * (landsat.vegetation_type == "tree"), -.2, 1.0)
landsat["source"] = "Landsat C2 L2 SR harmonized to Sentinel-2"
keep = []
for _, lr in landsat.iterrows():
    dates = S2[(S2.patch_id == lr.patch_id) & (S2.year == lr.year)].date
    keep.append(dates.empty or int((dates - lr.date).abs().dt.days.min()) > P["landsat_s2_calibration_window_days"])
fill = landsat[np.array(keep)].copy(); fill.to_csv(ROOT / "data/processed/landsat_temporal_fill.csv", index=False)
s2cols = ["patch_id", "block_id", "vegetation_type", "date", "year", "item_id", "valid_sample_points", "ndvi", "source"]
combined = pd.concat([S2[s2cols], fill[s2cols]], ignore_index=True).sort_values(["patch_id", "date"])
combined.to_csv(ROOT / "data/processed/combined_patch_timeseries.csv", index=False)
def density(series, label):
    rows = []
    for _, patch in PATCHES.iterrows():
        for year in P["years"]:
            g = series[(series.patch_id == patch.patch_id) & (series.year == year)]
            dates = sorted(pd.to_datetime(g.date).dt.date.unique()) if len(g) else []
            active = [d for d in dates if d.month in P["active_months"]]
            gaps = [(b - a).days for a, b in zip(active, active[1:])]
            summer = sum(d.month in P["summer_months"] for d in dates)
            rows.append({"series": label, "patch_id": patch.patch_id, "block_id": patch.block_id,
                "year": year, "vegetation_type": patch.vegetation_type,
                "urbanization_stratum": STRATUM.get(patch.block_id),
                "n_valid_observations": len(dates), "max_active_season_gap_days": max(gaps) if gaps else np.nan,
                "summer_observations": summer,
                "passes_temporal_qc": len(dates) >= P["minimum_patch_year_observations"] and
                    summer >= P["minimum_summer_observations"] and bool(gaps) and max(gaps) <= P["maximum_active_season_gap_days"]})
    return rows
density_df = pd.DataFrame(density(S2, "S2-only") + density(combined, "S2+Landsat-fill"))
density_df.to_csv(OUT / "observation_density.csv", index=False, encoding="utf-8-sig")
(ROOT / "logs/landsat_fill.log").write_text("\n".join(LOG), encoding="utf-8")
print({"landsat_patch_dates": len(landsat), "calibration_pairs": len(pairs), "pooled_rmse": rmse,
       "fill_dates": len(fill), "flagged_calibration_years": int(pd.DataFrame(annual).calibration_instability_flag.sum())})
print(pd.DataFrame(annual).to_string(index=False))
